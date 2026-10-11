// Copyright 2026 The trader Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command trader simulates a distributed next-byte model.
//
// There are N transformers. Each is two pre-norm causal layers with rotary
// positions and a byte embedding tied to the output projection. The corpus
// is pg100.txt. Works are the plays and poems named in the contents. Each
// transformer is given several consecutive windows from a single work.
//
// A window's first bytes are the training span. Its tail is held out and is
// the only loss used to decide a trade. The training forward stops at the
// end of that span, so the tail is not an input. Each cycle trains every
// node on its own windows, then builds a random perfect matching. A pair
// exchanges the window each model scores worst when that exchange lowers
// the sum of the two held-out losses. Both models then train on the window
// they received. The multiset of windows is unchanged.
//
// Build and run with the simd experiment:
//
//	GOEXPERIMENT=simd go run .
package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var errConfig = errors.New("invalid transformer config")

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "trader: %s\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("trader", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	file := fs.String("file", "pg100.txt", "corpus file")
	nodesN := fs.Int("nodes", 1024, "transformers")
	context := fs.Int("context", 1024, "bytes per window and context length")
	holdout := fs.Int("holdout", 256, "trailing bytes scored and withheld from training")
	perNode := fs.Int("windows", 4, "windows each node starts with, all from one work")
	steps := fs.Int("steps", 4, "Adam steps on a node's own windows each cycle")
	cycles := fs.Int("cycles", 4, "train-then-match cycles")
	adopt := fs.Int("adopt", 2, "Adam steps on a received window after a swap")
	lr := fs.Float64("lr", 3e-3, "Adam learning rate")
	seed := fs.Int64("seed", 1, "rng seed")
	workers := fs.Int("workers", 0, "workers (0 uses GOMAXPROCS)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected argument %q", fs.Arg(0))
	}
	if *nodesN < 2 {
		return fmt.Errorf("-nodes must be at least 2")
	}
	if *perNode < 2 {
		return fmt.Errorf("-windows must be at least 2")
	}
	if *steps < 0 || *cycles < 0 || *adopt < 0 {
		return fmt.Errorf("-steps, -cycles, and -adopt must be non-negative")
	}
	if !(*lr > 0) {
		return fmt.Errorf("-lr must be positive")
	}
	if *holdout < 1 || *holdout >= *context-1 {
		return fmt.Errorf("-holdout must leave a training prefix and a scored tail")
	}
	cfg := simConfig(*context)
	if err := cfg.validate(); err != nil {
		return fmt.Errorf("context %d is below 2", *context)
	}
	trainHi := *context - *holdout
	holdLo := trainHi
	nw := *workers
	if nw < 1 {
		nw = runtime.GOMAXPROCS(0)
	}

	data, err := os.ReadFile(*file)
	if err != nil {
		return err
	}
	works, err := parseWorks(data)
	if err != nil {
		return err
	}
	nodes, err := assignNodes(data, works, *nodesN, cfg.Context, *perNode, *seed)
	if err != nil {
		return err
	}
	plays := map[string]struct{}{}
	for i := range nodes {
		plays[nodes[i].origin] = struct{}{}
	}
	for i := range nodes {
		w := newTensors(cfg)
		initTensors(w, rand.New(rand.NewPCG(uint64(*seed)+uint64(i)*0x9e3779b97f4a7c15, uint64(i)+1)))
		nodes[i].w = w
		nodes[i].opt = newAdam(w)
		nodes[i].rng = rand.New(rand.NewPCG(uint64(*seed)+uint64(i)*0xbf58476d1ce4e5b9, uint64(i)+3))
	}

	say("corpus %s bytes=%d works=%d\n", *file, len(data), len(works))
	say("nodes=%d context=%d windows=%d train=%d holdout=%d layers=%d d=%d heads=%d dff=%d params=%d workers=%d plays=%d\n",
		len(nodes), cfg.Context, *perNode, trainHi, *holdout, cfg.Layers, cfg.D, cfg.Heads, cfg.DFF, paramCount(nodes[0].w), nw, len(plays))
	say("rotary positions, tied byte embedding\n")

	if err := scoreAll(nodes, cfg, holdLo, cfg.Context, nw, true); err != nil {
		return err
	}
	initMean, _, _, _, initAcc := lossStats(nodes)
	say("init    holdout mean=%.4f nats (%.3f bits) acc=%.4f\n", initMean, initMean/math.Ln2, initAcc)

	var trainedMean float64
	var traded int
	rng := rand.New(rand.NewPCG(uint64(*seed), 0x6a09e667f3bcc909))
	for c := 0; c < *cycles; c++ {
		t0 := time.Now()
		if *steps > 0 {
			if err := trainAll(nodes, cfg, *steps, 1, trainHi, float32(*lr), nw); err != nil {
				return err
			}
		}
		if err := scoreAll(nodes, cfg, holdLo, cfg.Context, nw, false); err != nil {
			return err
		}
		if c == 0 {
			trainedMean, _, _, _, _ = lossStats(nodes)
		}
		swaps, pairs, err := matchAll(nodes, cfg, rng, *adopt, 1, trainHi, holdLo, cfg.Context, float32(*lr), nw)
		if err != nil {
			return err
		}
		traded += swaps
		mean, _, _, _, acc := lossStats(nodes)
		say("cycle %d/%d holdout mean=%.4f acc=%.4f swaps=%d/%d %s\n",
			c+1, *cycles, mean, acc, swaps, pairs, time.Since(t0).Round(time.Millisecond))
	}
	if *cycles == 0 && *steps > 0 {
		if err := trainAll(nodes, cfg, *steps, 1, trainHi, float32(*lr), nw); err != nil {
			return err
		}
		if err := scoreAll(nodes, cfg, holdLo, cfg.Context, nw, false); err != nil {
			return err
		}
		trainedMean, _, _, _, _ = lossStats(nodes)
	}

	postMean, postMin, postMax, postAcc := currentStats(nodes)
	say("trained holdout mean=%.4f\n", trainedMean)
	say("post    holdout mean=%.4f min=%.4f max=%.4f acc=%.4f swaps=%d\n",
		postMean, postMin, postMax, postAcc, traded)
	nChunks, err := checkChunks(nodes)
	if err != nil {
		return err
	}
	say("unique windows=%d\n", nChunks)
	printSamples(nodes)

	ln := math.Log(float64(cfg.Vocab))
	if math.Abs(initMean-ln) > 1 {
		return fmt.Errorf("verification failed: initial holdout %.3f is far from ln(%d)=%.3f", initMean, cfg.Vocab, ln)
	}
	if *steps > 0 && !(trainedMean < initMean-0.05) {
		return fmt.Errorf("verification failed: holdout %.4f -> %.4f did not fall by 0.05 nats", initMean, trainedMean)
	}
	for i := range nodes {
		if math.IsNaN(nodes[i].own) || math.IsInf(nodes[i].own, 0) {
			return fmt.Errorf("verification failed: node %d loss is non-finite", i)
		}
	}
	say("verified\n")
	return nil
}

func say(format string, args ...any) {
	fmt.Printf(format, args...)
	os.Stdout.Sync()
}

type chunk struct {
	id   int
	work string
	off  int
	text []byte
	hold float64
	acc  float64
}

type node struct {
	id     int
	origin string
	bag    []chunk
	w      *tensors
	opt    *adam
	rng    *rand.Rand
	base   float64
	own    float64
	acc    float64
	trades int
}

type work struct {
	name       string
	start, end int
}

func parseWorks(data []byte) ([]work, error) {
	lines := splitLines(data)
	tocEnd := -1
	var titles []string
	inTOC := false
	for _, ln := range lines {
		trim := strings.TrimSpace(ln.text)
		if !inTOC && trim == "Contents" && len(titles) == 0 {
			inTOC = true
			continue
		}
		if !inTOC {
			continue
		}
		if trim == "" {
			if len(titles) > 0 {
				tocEnd = ln.start
				break
			}
			continue
		}
		titles = append(titles, trim)
	}
	if len(titles) < 2 || tocEnd < 0 {
		return nil, fmt.Errorf("contents list not found")
	}
	var found []work
	for _, title := range titles {
		off := -1
		for _, ln := range lines {
			if ln.start < tocEnd {
				continue
			}
			if ln.text == title {
				off = ln.start
				break
			}
		}
		if off < 0 {
			return nil, fmt.Errorf("work %q is not in the body", title)
		}
		found = append(found, work{name: title, start: off})
	}
	sort.Slice(found, func(i, j int) bool { return found[i].start < found[j].start })
	limit := len(data)
	if i := strings.Index(string(data), "\n*** END OF THE PROJECT GUTENBERG"); i >= 0 {
		limit = i + 1
	}
	for i := range found {
		if i+1 < len(found) {
			found[i].end = found[i+1].start
		} else {
			found[i].end = limit
		}
		if found[i].end < found[i].start {
			return nil, fmt.Errorf("work %q has a bad span", found[i].name)
		}
	}
	return found, nil
}

type fileLine struct {
	start int
	text  string
}

func splitLines(data []byte) []fileLine {
	var lines []fileLine
	i := 0
	for i < len(data) {
		j := i
		for j < len(data) && data[j] != '\n' {
			j++
		}
		text := string(data[i:j])
		text = strings.TrimRight(text, "\r")
		lines = append(lines, fileLine{start: i, text: text})
		if j < len(data) {
			j++
		}
		i = j
	}
	return lines
}

// sceneStart is the first ACT or SCENE line in the work, so a play's
// contents page is not the text the model fits. Poems stay at the title.
func sceneStart(data []byte, start, end int) int {
	lines := splitLines(data[start:end])
	for _, ln := range lines {
		trim := strings.TrimSpace(ln.text)
		if strings.HasPrefix(trim, "ACT ") || strings.HasPrefix(trim, "SCENE ") {
			return start + ln.start
		}
	}
	return start
}

func assignNodes(data []byte, works []work, nodes, context, perNode int, seed int64) ([]node, error) {
	type pool struct {
		name string
		offs []int
	}
	var pools []pool
	for _, w := range works {
		body := sceneStart(data, w.start, w.end)
		var offs []int
		for off := body; off+context <= w.end; off += context {
			offs = append(offs, off)
		}
		if len(offs) >= perNode {
			pools = append(pools, pool{w.name, offs})
		}
	}
	caps := make([]int, len(pools))
	weights := make([]int, len(pools))
	for i, p := range pools {
		caps[i] = len(p.offs) / perNode
		weights[i] = len(p.offs)
	}
	alloc, err := allocateNodes(weights, caps, nodes)
	if err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewPCG(uint64(seed), 99))
	var out []node
	nextID := 0
	for i, p := range pools {
		n := alloc[i]
		if n == 0 {
			continue
		}
		offs := append([]int(nil), p.offs...)
		rng.Shuffle(len(offs), func(a, b int) { offs[a], offs[b] = offs[b], offs[a] })
		use := n * perNode
		offs = offs[:use]
		sort.Ints(offs)
		for k := 0; k < n; k++ {
			bag := make([]chunk, perNode)
			for j := 0; j < perNode; j++ {
				off := offs[k*perNode+j]
				bag[j] = chunk{
					id:   nextID,
					work: p.name,
					off:  off,
					text: data[off : off+context],
				}
				nextID++
			}
			out = append(out, node{id: len(out), origin: p.name, bag: bag})
		}
	}
	if len(out) != nodes {
		return nil, fmt.Errorf("assigned %d nodes, need %d", len(out), nodes)
	}
	return out, nil
}

// allocateNodes spreads `nodes` seats across works, proportional to weight,
// never past cap. A cap of zero gets no seats.
func allocateNodes(weight, cap []int, nodes int) ([]int, error) {
	if len(weight) != len(cap) {
		return nil, fmt.Errorf("allocation shape")
	}
	totalCap := 0
	sumW := 0
	for i := range weight {
		if cap[i] < 0 || weight[i] < 0 {
			return nil, fmt.Errorf("allocation shape")
		}
		totalCap += cap[i]
		if cap[i] > 0 {
			sumW += weight[i]
		}
	}
	if totalCap < nodes {
		return nil, fmt.Errorf("only %d nodes can each hold the requested windows from one work, need %d", totalCap, nodes)
	}
	alloc := make([]int, len(weight))
	assigned := 0
	type rem struct {
		i int
		f float64
	}
	var order []rem
	for i := range weight {
		if cap[i] == 0 || sumW == 0 {
			continue
		}
		exact := float64(nodes) * float64(weight[i]) / float64(sumW)
		n := int(exact)
		if n > cap[i] {
			n = cap[i]
		}
		alloc[i] = n
		assigned += n
		order = append(order, rem{i, exact - float64(n)})
	}
	sort.Slice(order, func(a, b int) bool { return order[a].f > order[b].f })
	for assigned < nodes {
		moved := false
		for _, r := range order {
			if alloc[r.i] < cap[r.i] {
				alloc[r.i]++
				assigned++
				moved = true
				if assigned == nodes {
					break
				}
			}
		}
		if !moved {
			return nil, fmt.Errorf("could not assign %d nodes", nodes)
		}
	}
	return alloc, nil
}

func worstIndex(n *node) int {
	bi := 0
	for i := 1; i < len(n.bag); i++ {
		if n.bag[i].hold > n.bag[bi].hold {
			bi = i
		}
	}
	return bi
}

func refreshOwn(n *node) {
	var sum, acc float64
	for _, c := range n.bag {
		sum += c.hold
		acc += c.acc
	}
	n.own = sum / float64(len(n.bag))
	n.acc = acc / float64(len(n.bag))
}

func finiteLoss(cfg Config, w *tensors, ws *workspace, text []byte, tgtLo, tgtHi int) (float64, float64, error) {
	lo, hi := lossSpan(len(text), tgtLo, tgtHi)
	if hi <= lo {
		return 0, 0, fmt.Errorf("loss span")
	}
	loss, correct := forwardBackward(cfg, w, nil, ws, text, tgtLo, tgtHi)
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return 0, 0, fmt.Errorf("non-finite loss")
	}
	return loss, float64(correct) / float64(hi-lo), nil
}

func learnStep(cfg Config, w, g *tensors, opt *adam, ws *workspace, text []byte, tgtLo, tgtHi int, lr float32) (float64, error) {
	loss, _ := forwardBackward(cfg, w, g, ws, text, tgtLo, tgtHi)
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return 0, fmt.Errorf("non-finite loss")
	}
	norm := clipGrads(g, 1)
	if math.IsNaN(norm) || math.IsInf(norm, 0) {
		return 0, fmt.Errorf("non-finite gradient")
	}
	opt.step(w, g, lr)
	return loss, nil
}

func scoreNode(n *node, cfg Config, ws *workspace, tgtLo, tgtHi int) error {
	for i := range n.bag {
		loss, acc, err := finiteLoss(cfg, n.w, ws, n.bag[i].text, tgtLo, tgtHi)
		if err != nil {
			return err
		}
		n.bag[i].hold = loss
		n.bag[i].acc = acc
	}
	refreshOwn(n)
	return nil
}

func parallel(n, workers int, cfg Config, grad bool, fn func(i int, ws *workspace, g *tensors) error) error {
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}
	var next atomic.Int64
	errc := make(chan error, workers)
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			ws := newWorkspace(cfg)
			var g *tensors
			if grad {
				g = newTensors(cfg)
			}
			for {
				i := int(next.Add(1) - 1)
				if i >= n {
					return
				}
				if err := fn(i, ws, g); err != nil {
					errc <- err
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errc)
	for err := range errc {
		if err != nil {
			return err
		}
	}
	return nil
}

func scoreAll(nodes []node, cfg Config, tgtLo, tgtHi, workers int, init bool) error {
	err := parallel(len(nodes), workers, cfg, false, func(i int, ws *workspace, _ *tensors) error {
		if err := scoreNode(&nodes[i], cfg, ws, tgtLo, tgtHi); err != nil {
			return fmt.Errorf("node %d: %w", i, err)
		}
		if init {
			nodes[i].base = nodes[i].own
		}
		return nil
	})
	return err
}

// trainAll takes steps Adam updates on each node. An update reads one
// random window and stops at tgtHi, so a held-out tail is not an input.
func trainAll(nodes []node, cfg Config, steps, tgtLo, tgtHi int, lr float32, workers int) error {
	return parallel(len(nodes), workers, cfg, true, func(i int, ws *workspace, g *tensors) error {
		n := &nodes[i]
		for s := 0; s < steps; s++ {
			c := &n.bag[n.rng.IntN(len(n.bag))]
			if _, err := learnStep(cfg, n.w, g, n.opt, ws, c.text, tgtLo, tgtHi, lr); err != nil {
				return fmt.Errorf("node %d: %w", n.id, err)
			}
		}
		return nil
	})
}

type pair struct {
	a, b       int
	ia, ib     int
	ownA, ownB float64
	crossA     float64
	crossB     float64
	swap       bool
}

func planMatch(nodes []node, rng *rand.Rand) []pair {
	order := make([]int, len(nodes))
	for i := range order {
		order[i] = i
	}
	rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	var pairs []pair
	for i := 0; i+1 < len(order); i += 2 {
		a, b := order[i], order[i+1]
		ia, ib := worstIndex(&nodes[a]), worstIndex(&nodes[b])
		pairs = append(pairs, pair{
			a: a, b: b, ia: ia, ib: ib,
			ownA: nodes[a].bag[ia].hold,
			ownB: nodes[b].bag[ib].hold,
		})
	}
	return pairs
}

// gainsSwap is true when exchanging the two windows lowers the sum of their
// held-out losses.
func gainsSwap(ownA, crossA, ownB, crossB float64) bool {
	return crossA+crossB < ownA+ownB
}

func matchAll(nodes []node, cfg Config, rng *rand.Rand, adopt, trainLo, trainHi, holdLo, holdHi int, lr float32, workers int) (int, int, error) {
	pairs := planMatch(nodes, rng)
	if err := parallel(len(pairs), workers, cfg, false, func(i int, ws *workspace, _ *tensors) error {
		p := &pairs[i]
		wa := nodes[p.a].bag[p.ia].text
		wb := nodes[p.b].bag[p.ib].text
		ca, _, err := finiteLoss(cfg, nodes[p.a].w, ws, wb, holdLo, holdHi)
		if err != nil {
			return err
		}
		cb, _, err := finiteLoss(cfg, nodes[p.b].w, ws, wa, holdLo, holdHi)
		if err != nil {
			return err
		}
		p.crossA = ca
		p.crossB = cb
		p.swap = gainsSwap(p.ownA, ca, p.ownB, cb)
		return nil
	}); err != nil {
		return 0, 0, err
	}
	swaps := 0
	swapped := make([]bool, len(nodes))
	for i := range pairs {
		p := &pairs[i]
		if !p.swap {
			continue
		}
		a, b := &nodes[p.a], &nodes[p.b]
		a.bag[p.ia], b.bag[p.ib] = b.bag[p.ib], a.bag[p.ia]
		a.trades++
		b.trades++
		swapped[p.a] = true
		swapped[p.b] = true
		swaps++
	}
	if adopt > 0 && swaps > 0 {
		if err := parallel(len(nodes), workers, cfg, true, func(i int, ws *workspace, g *tensors) error {
			if !swapped[i] {
				return nil
			}
			n := &nodes[i]
			// The received window is the one just written into the bag slot
			// that this node offered. Both endpoints of a pair were flagged.
			var text []byte
			for _, p := range pairs {
				if !p.swap {
					continue
				}
				if p.a == i {
					text = n.bag[p.ia].text
					break
				}
				if p.b == i {
					text = n.bag[p.ib].text
					break
				}
			}
			for s := 0; s < adopt; s++ {
				if _, err := learnStep(cfg, n.w, g, n.opt, ws, text, trainLo, trainHi, lr); err != nil {
					return fmt.Errorf("node %d: %w", n.id, err)
				}
			}
			return scoreNode(n, cfg, ws, holdLo, holdHi)
		}); err != nil {
			return swaps, len(pairs), err
		}
	} else if swaps > 0 {
		if err := parallel(len(nodes), workers, cfg, false, func(i int, ws *workspace, _ *tensors) error {
			if !swapped[i] {
				return nil
			}
			return scoreNode(&nodes[i], cfg, ws, holdLo, holdHi)
		}); err != nil {
			return swaps, len(pairs), err
		}
	}
	return swaps, len(pairs), nil
}

// exchange is the single-pair form of a match, used by tests.
func exchange(a, b *node, cfg Config, ws *workspace, g *tensors, adopt, trainLo, trainHi, holdLo, holdHi int, lr float32) (bool, error) {
	ia, ib := worstIndex(a), worstIndex(b)
	ownA, ownB := a.bag[ia].hold, b.bag[ib].hold
	crossA, _, err := finiteLoss(cfg, a.w, ws, b.bag[ib].text, holdLo, holdHi)
	if err != nil {
		return false, err
	}
	crossB, _, err := finiteLoss(cfg, b.w, ws, a.bag[ia].text, holdLo, holdHi)
	if err != nil {
		return false, err
	}
	if !gainsSwap(ownA, crossA, ownB, crossB) {
		return false, nil
	}
	a.bag[ia], b.bag[ib] = b.bag[ib], a.bag[ia]
	for s := 0; s < adopt; s++ {
		if _, err = learnStep(cfg, a.w, g, a.opt, ws, a.bag[ia].text, trainLo, trainHi, lr); err != nil {
			return false, err
		}
		if _, err = learnStep(cfg, b.w, g, b.opt, ws, b.bag[ib].text, trainLo, trainHi, lr); err != nil {
			return false, err
		}
	}
	if err = scoreNode(a, cfg, ws, holdLo, holdHi); err != nil {
		return false, err
	}
	if err = scoreNode(b, cfg, ws, holdLo, holdHi); err != nil {
		return false, err
	}
	a.trades++
	b.trades++
	return true, nil
}

func lossStats(nodes []node) (mean, min, max, base, acc float64) {
	min = math.Inf(1)
	for i := range nodes {
		mean += nodes[i].own
		base += nodes[i].base
		acc += nodes[i].acc
		if nodes[i].own < min {
			min = nodes[i].own
		}
		if nodes[i].own > max {
			max = nodes[i].own
		}
	}
	n := float64(len(nodes))
	return mean / n, min, max, base / n, acc / n
}

func currentStats(nodes []node) (mean, min, max, acc float64) {
	mean, min, max, _, acc = lossStats(nodes)
	return mean, min, max, acc
}

func checkChunks(nodes []node) (int, error) {
	seen := make(map[int]int)
	n := 0
	for i := range nodes {
		if len(nodes[i].bag) < 2 {
			return 0, fmt.Errorf("verification failed: node %d holds %d windows", i, len(nodes[i].bag))
		}
		for _, c := range nodes[i].bag {
			if len(c.text) == 0 {
				return 0, fmt.Errorf("verification failed: empty window %d", c.id)
			}
			seen[c.id]++
			n++
		}
	}
	if len(seen) != n {
		return 0, fmt.Errorf("verification failed: %d windows but %d distinct ids", n, len(seen))
	}
	for id := 0; id < n; id++ {
		if seen[id] != 1 {
			return 0, fmt.Errorf("verification failed: window %d is held by %d nodes", id, seen[id])
		}
	}
	return n, nil
}

func printSamples(nodes []node) {
	idxs := []int{0, len(nodes) / 4, len(nodes) / 2, 3 * len(nodes) / 4}
	seen := map[int]bool{}
	say("%-6s %-6s %-8s %-8s %s\n", "node", "bag", "trades", "loss", "origin / text")
	for _, id := range idxs {
		if id < 0 || id >= len(nodes) || seen[id] {
			continue
		}
		seen[id] = true
		n := &nodes[id]
		say("%-6d %-6d %-8d %-8.3f %s | %s\n",
			n.id, len(n.bag), n.trades, n.own, shortName(n.origin), preview(n.bag[0].text))
	}
	best := 0
	for i := range nodes {
		if nodes[i].trades > nodes[best].trades {
			best = i
		}
	}
	if nodes[best].trades > 0 {
		n := &nodes[best]
		say("most trades node %d count=%d loss=%.3f %s | %s\n",
			n.id, n.trades, n.own, shortName(n.origin), preview(n.bag[0].text))
	}
}

func shortName(s string) string {
	const n = 32
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}

func preview(b []byte) string {
	n := 42
	if len(b) < n {
		n = len(b)
	}
	buf := make([]byte, n)
	for i := 0; i < n; i++ {
		c := b[i]
		if c < 32 || c > 126 {
			c = ' '
		}
		buf[i] = c
	}
	return string(buf)
}
