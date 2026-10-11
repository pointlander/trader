// Copyright 2026 The trader Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command trader simulates a distributed next-byte model.
//
// There are N transformers. Each is one pre-norm causal layer with context
// length 1024. The corpus is pg100.txt. N non-overlapping 1024-byte windows
// are spread across the file, one per transformer. Each transformer is
// trained on its own window.
//
// Trading then draws random pairs. A pair exchanges windows when each model
// has a lower next-byte loss on the other's window than on its own. Both
// models train on the window they receive, and that window becomes the text
// they hold. A pair that fails the test keeps its windows. The multiset of
// windows is unchanged; only the owners move.
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
	nodesN := fs.Int("nodes", 1024, "transformers, and windows assigned to them")
	context := fs.Int("context", 1024, "bytes per window and context length")
	steps := fs.Int("steps", 8, "Adam steps on a node's own window before trading")
	rounds := fs.Int("rounds", 1024, "random pairs drawn during trading")
	adopt := fs.Int("adopt", 4, "Adam steps on a received window after an accepted trade")
	lr := fs.Float64("lr", 3e-3, "Adam learning rate")
	seed := fs.Int64("seed", 1, "rng seed")
	workers := fs.Int("workers", 0, "training workers (0 uses GOMAXPROCS)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected argument %q", fs.Arg(0))
	}
	if *nodesN < 1 {
		return fmt.Errorf("-nodes must be positive")
	}
	if *steps < 0 || *rounds < 0 || *adopt < 0 {
		return fmt.Errorf("-steps, -rounds, and -adopt must be non-negative")
	}
	if !(*lr > 0) {
		return fmt.Errorf("-lr must be positive")
	}
	cfg := simConfig(*context)
	if err := cfg.validate(); err != nil {
		return fmt.Errorf("context %d is below 2", *context)
	}
	nw := *workers
	if nw < 1 {
		nw = runtime.GOMAXPROCS(0)
	}

	data, err := os.ReadFile(*file)
	if err != nil {
		return err
	}
	starts, err := windowStarts(len(data), cfg.Context, *nodesN)
	if err != nil {
		return err
	}

	nodes := make([]node, *nodesN)
	for i := range nodes {
		w := newTensors(cfg)
		initTensors(w, rand.New(rand.NewPCG(uint64(*seed)+uint64(i)*0x9e3779b97f4a7c15, uint64(i)+1)))
		nodes[i] = node{
			id:   i,
			src:  i,
			text: data[starts[i] : starts[i]+cfg.Context],
			w:    w,
			opt:  newAdam(w),
		}
	}
	fmt.Printf("corpus %s bytes=%d\n", *file, len(data))
	fmt.Printf("nodes=%d context=%d d=%d heads=%d dff=%d params=%d workers=%d\n",
		len(nodes), cfg.Context, cfg.D, cfg.Heads, cfg.DFF, paramCount(nodes[0].w), nw)
	fmt.Printf("windows spread across the file, offsets %d..%d\n", starts[0], starts[len(starts)-1])

	tTrain := time.Now()
	if err := trainNodes(nodes, cfg, *steps, float32(*lr), nw); err != nil {
		return err
	}
	fmt.Printf("trained in %s\n", time.Since(tTrain).Round(time.Millisecond))
	initMean, trainMean, trainMin, trainMax, trainAcc := lossStats(nodes)
	fmt.Printf("init    mean=%.4f nats (%.3f bits)\n", initMean, initMean/math.Ln2)
	fmt.Printf("trained mean=%.4f min=%.4f max=%.4f acc=%.4f\n", trainMean, trainMin, trainMax, trainAcc)

	rng := rand.New(rand.NewPCG(uint64(*seed), 0x6a09e667f3bcc909))
	var traded, oneSided, neither int
	var ownSum, crossSum float64
	var scored int
	tTrade := time.Now()
	if *rounds > 0 && len(nodes) >= 2 {
		ws := newWorkspace(cfg)
		g := newTensors(cfg)
		every := *rounds / 16
		if every < 1 {
			every = 1
		}
		for r := 0; r < *rounds; r++ {
			i := rng.IntN(len(nodes))
			j := rng.IntN(len(nodes) - 1)
			if j >= i {
				j++
			}
			outcome, ownI, crossI, ownJ, crossJ, err := considerPair(&nodes[i], &nodes[j], cfg, ws, g, *adopt, float32(*lr))
			if err != nil {
				return err
			}
			ownSum += ownI + ownJ
			crossSum += crossI + crossJ
			scored += 2
			switch outcome {
			case pairTrade:
				traded++
			case pairOne:
				oneSided++
			default:
				neither++
			}
			if (r+1)%every == 0 || r+1 == *rounds {
				fmt.Printf("round %4d/%d traded=%d one-sided=%d neither=%d %s\n",
					r+1, *rounds, traded, oneSided, neither, time.Since(tTrade).Round(time.Millisecond))
			}
		}
	} else {
		fmt.Println("trading skipped")
	}
	fmt.Printf("traded in %s\n", time.Since(tTrade).Round(time.Millisecond))

	postMean, postMin, postMax := currentLoss(nodes)
	fmt.Printf("pairs=%d traded=%d one-sided=%d neither=%d\n", traded+oneSided+neither, traded, oneSided, neither)
	if scored > 0 {
		fmt.Printf("pair mean own=%.4f traded=%.4f\n", ownSum/float64(scored), crossSum/float64(scored))
		fmt.Printf("traded text had lower loss in %d/%d comparisons\n", oneSided+2*traded, scored)
	}
	fmt.Printf("post    mean=%.4f min=%.4f max=%.4f\n", postMean, postMin, postMax)
	if err := checkPermutation(nodes); err != nil {
		return err
	}
	fmt.Printf("unique windows=%d\n", len(nodes))
	printSamples(nodes, starts)

	ln := math.Log(float64(cfg.Vocab))
	if math.Abs(initMean-ln) > 1 {
		return fmt.Errorf("verification failed: initial loss %.3f is far from ln(%d)=%.3f", initMean, cfg.Vocab, ln)
	}
	if *steps > 0 && !(trainMean < initMean-0.25) {
		return fmt.Errorf("verification failed: mean loss %.4f -> %.4f did not fall by 0.25 nats", initMean, trainMean)
	}
	for i := range nodes {
		if math.IsNaN(nodes[i].own) || math.IsInf(nodes[i].own, 0) {
			return fmt.Errorf("verification failed: node %d loss is non-finite", i)
		}
	}
	fmt.Println("verified")
	return nil
}

type node struct {
	id     int
	src    int
	text   []byte
	w      *tensors
	opt    *adam
	base   float64
	own    float64
	acc    float64
	trades int
}

const (
	pairNeither = 0
	pairOne     = 1
	pairTrade   = 2
)

// considerPair trades windows when both models score the other's text lower
// than their own. The returned own and cross losses are the pre-trade values
// the test used. After an accepted trade, each node trains on the window it
// received and own is refreshed.
func considerPair(a, b *node, cfg Config, ws *workspace, g *tensors, adopt int, lr float32) (outcome int, ownA, crossA, ownB, crossB float64, err error) {
	ownA, ownB = a.own, b.own
	crossA, _, err = finiteLoss(cfg, a.w, ws, b.text)
	if err != nil {
		return 0, 0, 0, 0, 0, err
	}
	crossB, _, err = finiteLoss(cfg, b.w, ws, a.text)
	if err != nil {
		return 0, 0, 0, 0, 0, err
	}
	aWants := crossA < ownA
	bWants := crossB < ownB
	if !(aWants && bWants) {
		if aWants || bWants {
			return pairOne, ownA, crossA, ownB, crossB, nil
		}
		return pairNeither, ownA, crossA, ownB, crossB, nil
	}
	a.text, b.text = b.text, a.text
	a.src, b.src = b.src, a.src
	for s := 0; s < adopt; s++ {
		if _, err = learnStep(cfg, a.w, g, a.opt, ws, a.text, lr); err != nil {
			return 0, 0, 0, 0, 0, err
		}
		if _, err = learnStep(cfg, b.w, g, b.opt, ws, b.text, lr); err != nil {
			return 0, 0, 0, 0, 0, err
		}
	}
	if a.own, a.acc, err = finiteLoss(cfg, a.w, ws, a.text); err != nil {
		return 0, 0, 0, 0, 0, err
	}
	if b.own, b.acc, err = finiteLoss(cfg, b.w, ws, b.text); err != nil {
		return 0, 0, 0, 0, 0, err
	}
	a.trades++
	b.trades++
	return pairTrade, ownA, crossA, ownB, crossB, nil
}

func finiteLoss(cfg Config, w *tensors, ws *workspace, text []byte) (float64, float64, error) {
	loss, correct := forwardBackward(cfg, w, nil, ws, text)
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return 0, 0, fmt.Errorf("non-finite loss")
	}
	acc := float64(correct) / float64(cfg.preds())
	return loss, acc, nil
}

func learnStep(cfg Config, w, g *tensors, opt *adam, ws *workspace, text []byte, lr float32) (float64, error) {
	loss, _ := forwardBackward(cfg, w, g, ws, text)
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

func trainNodes(nodes []node, cfg Config, steps int, lr float32, workers int) error {
	var next atomic.Int64
	var mu sync.Mutex
	done := 0
	errc := make(chan error, workers)
	start := time.Now()
	stop := make(chan struct{})
	go func() {
		t := time.NewTicker(2 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				mu.Lock()
				d := done
				mu.Unlock()
				if d >= len(nodes) {
					return
				}
				fmt.Printf("train %d/%d %s\n", d, len(nodes), time.Since(start).Round(time.Second))
			}
		}
	}()
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			ws := newWorkspace(cfg)
			g := newTensors(cfg)
			for {
				i := int(next.Add(1) - 1)
				if i >= len(nodes) {
					return
				}
				n := &nodes[i]
				base, _, err := finiteLoss(cfg, n.w, ws, n.text)
				if err != nil {
					errc <- fmt.Errorf("node %d: %w", n.id, err)
					return
				}
				for s := 0; s < steps; s++ {
					if _, err = learnStep(cfg, n.w, g, n.opt, ws, n.text, lr); err != nil {
						errc <- fmt.Errorf("node %d: %w", n.id, err)
						return
					}
				}
				own, acc, err := finiteLoss(cfg, n.w, ws, n.text)
				if err != nil {
					errc <- fmt.Errorf("node %d: %w", n.id, err)
					return
				}
				n.base = base
				n.own = own
				n.acc = acc
				mu.Lock()
				done++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	close(stop)
	close(errc)
	for err := range errc {
		if err != nil {
			return err
		}
	}
	return nil
}

func lossStats(nodes []node) (initMean, trainMean, trainMin, trainMax, trainAcc float64) {
	trainMin = math.Inf(1)
	for i := range nodes {
		initMean += nodes[i].base
		trainMean += nodes[i].own
		trainAcc += nodes[i].acc
		if nodes[i].own < trainMin {
			trainMin = nodes[i].own
		}
		if nodes[i].own > trainMax {
			trainMax = nodes[i].own
		}
	}
	n := float64(len(nodes))
	return initMean / n, trainMean / n, trainMin, trainMax, trainAcc / n
}

func currentLoss(nodes []node) (mean, min, max float64) {
	min = math.Inf(1)
	for i := range nodes {
		v := nodes[i].own
		mean += v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	mean /= float64(len(nodes))
	return mean, min, max
}

func checkPermutation(nodes []node) error {
	seen := make([]int, len(nodes))
	for i := range nodes {
		s := nodes[i].src
		if s < 0 || s >= len(nodes) || len(nodes[i].text) != len(nodes[0].text) {
			return fmt.Errorf("verification failed: node %d holds an invalid window", i)
		}
		seen[s]++
	}
	for s, c := range seen {
		if c != 1 {
			return fmt.Errorf("verification failed: window %d is held by %d nodes", s, c)
		}
	}
	return nil
}

func printSamples(nodes []node, starts []int) {
	idxs := []int{0, len(nodes) / 4, len(nodes) / 2, 3 * len(nodes) / 4}
	if len(nodes) == 1 {
		idxs = []int{0}
	}
	seen := map[int]bool{}
	fmt.Printf("%-6s %-6s %-8s %-8s %s\n", "node", "src", "offset", "loss", "text")
	for _, id := range idxs {
		if id < 0 || id >= len(nodes) || seen[id] {
			continue
		}
		seen[id] = true
		n := &nodes[id]
		fmt.Printf("%-6d %-6d %-8d %-8.3f %s\n", n.id, n.src, starts[n.src], n.own, preview(n.text))
	}
	best := 0
	for i := range nodes {
		if nodes[i].trades > nodes[best].trades {
			best = i
		}
	}
	if nodes[best].trades > 0 {
		n := &nodes[best]
		fmt.Printf("most trades node %d count=%d src=%d loss=%.3f %s\n",
			n.id, n.trades, n.src, n.own, preview(n.text))
	}
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

// windowStarts picks nodes non-overlapping windows of context bytes, spread
// across a file of n bytes. Window i begins at a multiple of context.
func windowStarts(n, context, nodes int) ([]int, error) {
	if context < 2 {
		return nil, fmt.Errorf("context %d is below 2", context)
	}
	if nodes < 1 {
		return nil, fmt.Errorf("nodes %d is below 1", nodes)
	}
	usable := n / context
	if nodes > usable {
		return nil, fmt.Errorf("file has %d non-overlapping windows of %d bytes, need %d", usable, context, nodes)
	}
	starts := make([]int, nodes)
	prev := -1
	for i := 0; i < nodes; i++ {
		off := i * usable / nodes
		if off <= prev {
			off = prev + 1
		}
		if off >= usable {
			return nil, fmt.Errorf("window %d does not fit", i)
		}
		starts[i] = off * context
		prev = off
	}
	return starts, nil
}
