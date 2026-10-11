// Copyright 2026 The trader Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"
)

func testConfig() Config {
	return Config{Vocab: 16, Context: 6, D: 8, Heads: 2, DFF: 16, Layers: 2}
}

func TestFiniteDifference(t *testing.T) {
	if err := (Config{Vocab: 8, Context: 4, D: 6, Heads: 2, DFF: 8, Layers: 1}).validate(); err == nil {
		t.Fatal("odd head width accepted")
	}
	cfg := testConfig()
	w := newTensors(cfg)
	initTensors(w, rand.New(rand.NewPCG(3, 1)))
	ws := newWorkspace(cfg)
	g := newTensors(cfg)
	text := make([]byte, cfg.Context)
	for i := range text {
		text[i] = byte((i*3 + 1) % cfg.Vocab)
	}
	forwardBackward(cfg, w, g, ws, text, 1, cfg.Context)

	type check struct {
		name string
		p, g []float32
		idx  []int
	}
	checks := []check{
		{"Emb", w.Emb, g.Emb, []int{0, cfg.D, 2 * cfg.D}},
		{"Ln1W", w.Layers[0].Ln1W, g.Layers[0].Ln1W, []int{0, cfg.D / 2, cfg.D - 1}},
		{"Wq", w.Layers[0].Wq, g.Layers[0].Wq, []int{0, len(w.Layers[0].Wq) / 2, len(w.Layers[0].Wq) - 1}},
		{"Wo", w.Layers[0].Wo, g.Layers[0].Wo, []int{1, len(w.Layers[0].Wo) / 3, len(w.Layers[0].Wo) - 2}},
		{"W1", w.Layers[0].W1, g.Layers[0].W1, []int{0, len(w.Layers[0].W1) / 2, len(w.Layers[0].W1) - 1}},
		{"W2", w.Layers[0].W2, g.Layers[0].W2, []int{0, 7, len(w.Layers[0].W2) - 1}},
		{"Wq1", w.Layers[1].Wq, g.Layers[1].Wq, []int{0, len(w.Layers[1].Wq) / 2, len(w.Layers[1].Wq) - 1}},
		{"W21", w.Layers[1].W2, g.Layers[1].W2, []int{0, len(w.Layers[1].W2) / 2, len(w.Layers[1].W2) - 1}},
		{"LnFW", w.LnFW, g.LnFW, []int{0, cfg.D - 1}},
		{"Bh", w.Bh, g.Bh, []int{0, int(text[1]), cfg.Vocab - 1}},
	}
	for _, c := range checks {
		for _, idx := range c.idx {
			fdCheck(t, c.name, c.p, c.g[idx], idx, cfg, w, ws, text)
		}
	}
}

func fdCheck(t *testing.T, label string, p []float32, ana float32, idx int, cfg Config, w *tensors, ws *workspace, text []byte) {
	t.Helper()
	const eps = float32(1e-3)
	p[idx] += eps
	lp, _ := forwardBackward(cfg, w, nil, ws, text, 1, cfg.Context)
	p[idx] -= 2 * eps
	lm, _ := forwardBackward(cfg, w, nil, ws, text, 1, cfg.Context)
	p[idx] += eps
	num := (lp - lm) / float64(2*eps)
	diff := math.Abs(num - float64(ana))
	rel := diff
	if math.Abs(num) > 1e-4 {
		rel = diff / math.Abs(num)
	}
	if diff > 1e-3 && rel > 0.05 {
		t.Fatalf("%s[%d] numeric=%g analytic=%g", label, idx, num, ana)
	}
}

func TestCausalMask(t *testing.T) {
	cfg := testConfig()
	w := newTensors(cfg)
	initTensors(w, rand.New(rand.NewPCG(4, 2)))
	ws := newWorkspace(cfg)
	text := []byte{1, 2, 3, 4, 5, 6}
	forwardBackward(cfg, w, nil, ws, text, 1, cfg.Context)
	first := append([]float32(nil), ws.logits[:cfg.Vocab*(cfg.Context-1)]...)
	text[len(text)-1] = 9
	forwardBackward(cfg, w, nil, ws, text, 1, cfg.Context)
	for i, v := range first {
		if ws.logits[i] != v {
			t.Fatalf("future byte changed logit %d", i)
		}
	}
}

func TestRopeRoundtrip(t *testing.T) {
	const T, D, H, Dh = 4, 8, 2, 4
	cos := make([]float32, T*(Dh/2))
	sin := make([]float32, T*(Dh/2))
	buildRope(cos, sin, T, Dh)
	x := make([]float32, T*D)
	for i := range x {
		x[i] = float32(i+1) * 0.01
	}
	orig := append([]float32(nil), x...)
	applyRope(x, T, D, H, Dh, cos, sin, 1)
	moved := false
	for i := range x {
		if x[i] != orig[i] {
			moved = true
			break
		}
	}
	if !moved {
		t.Fatal("rope left the vector unchanged")
	}
	applyRope(x, T, D, H, Dh, cos, sin, -1)
	for i := range x {
		if math.Abs(float64(x[i]-orig[i])) > 1e-5 {
			t.Fatalf("rope roundtrip at %d: got %g want %g", i, x[i], orig[i])
		}
	}
}

func TestPrefixIsolated(t *testing.T) {
	cfg := Config{Vocab: 32, Context: 8, D: 8, Heads: 2, DFF: 16, Layers: 2}
	w := newTensors(cfg)
	initTensors(w, rand.New(rand.NewPCG(11, 4)))
	ws := newWorkspace(cfg)
	g := newTensors(cfg)
	text := make([]byte, cfg.Context)
	for i := range text {
		text[i] = byte(i%5 + 1)
	}
	text[6] = 20
	text[7] = 21

	const trainHi = 6
	forwardBackward(cfg, w, nil, ws, text, 1, cfg.Context)
	full := append([]float32(nil), ws.logits[:trainHi*cfg.Vocab]...)
	forwardBackward(cfg, w, g, ws, text, 1, trainHi)
	if ws.T != trainHi {
		t.Fatalf("active length %d, want %d", ws.T, trainHi)
	}
	for i, v := range full {
		if ws.logits[i] != v {
			t.Fatalf("tail changed prefix logit %d", i)
		}
	}
	// Predictors of targets [1, trainHi) are positions [0, trainHi-1).
	// The last position predicts a byte past the span, so its logit grad is zero.
	for i := 0; i < trainHi; i++ {
		row := ws.glogits[i*cfg.Vocab : (i+1)*cfg.Vocab]
		nonzero := false
		for _, v := range row {
			if v != 0 {
				nonzero = true
				break
			}
		}
		want := i < trainHi-1
		if nonzero != want {
			t.Fatalf("position %d grad nonzero=%v", i, nonzero)
		}
	}

	loss1, _ := forwardBackward(cfg, w, nil, ws, text, 1, trainHi)
	text[7] ^= 3
	loss2, _ := forwardBackward(cfg, w, nil, ws, text, 1, trainHi)
	if loss1 != loss2 {
		t.Fatalf("held-out byte changed prefix loss %g %g", loss1, loss2)
	}
	text[7] ^= 3
	hold1, _ := forwardBackward(cfg, w, nil, ws, text, trainHi, cfg.Context)
	text[7] ^= 3
	hold2, _ := forwardBackward(cfg, w, nil, ws, text, trainHi, cfg.Context)
	if hold1 == hold2 {
		t.Fatal("held-out byte did not change held-out loss")
	}
}

func TestLearnsBigram(t *testing.T) {
	cfg := Config{Vocab: 256, Context: 16, D: 16, Heads: 2, DFF: 32, Layers: 2}
	w := newTensors(cfg)
	initTensors(w, rand.New(rand.NewPCG(5, 3)))
	opt := newAdam(w)
	ws := newWorkspace(cfg)
	g := newTensors(cfg)
	text := make([]byte, cfg.Context)
	pat := []byte("abcd")
	for i := range text {
		text[i] = pat[i%len(pat)]
	}
	before, _, err := finiteLoss(cfg, w, ws, text, 1, cfg.Context)
	if err != nil {
		t.Fatal(err)
	}
	for s := 0; s < 40; s++ {
		if _, err := learnStep(cfg, w, g, opt, ws, text, 1, cfg.Context, 1e-2); err != nil {
			t.Fatal(err)
		}
	}
	after, acc, err := finiteLoss(cfg, w, ws, text, 1, cfg.Context)
	if err != nil {
		t.Fatal(err)
	}
	if !(after < before-1) {
		t.Fatalf("loss %.3f -> %.3f, acc %.3f", before, after, acc)
	}
}

func TestExchange(t *testing.T) {
	if gainsSwap(1, 0.4, 1, 1.6) {
		t.Fatal("equal sums swapped")
	}
	if !gainsSwap(1, 0.4, 1, 1.5) {
		t.Fatal("a lower sum did not swap")
	}
	if gainsSwap(2, 3, 2, 2) {
		t.Fatal("a higher sum swapped")
	}

	cfg := Config{Vocab: 256, Context: 16, D: 8, Heads: 2, DFF: 16, Layers: 1}
	ws := newWorkspace(cfg)
	g := newTensors(cfg)
	mk := func(id int, seed uint64, bag []chunk) node {
		w := newTensors(cfg)
		initTensors(w, rand.New(rand.NewPCG(seed, 1)))
		return node{id: id, w: w, opt: newAdam(w), bag: bag}
	}
	pat := func(b byte) []byte { return bytes.Repeat([]byte{b}, cfg.Context) }
	const holdLo, holdHi = 12, 16

	// Owns equal the cross losses, so the sum does not fall.
	a := mk(0, 20, []chunk{{id: 0, text: pat('a'), hold: 0}, {id: 1, text: pat('b'), hold: -1e9}})
	b := mk(1, 21, []chunk{{id: 2, text: pat('c'), hold: 0}, {id: 3, text: pat('d'), hold: -1e9}})
	ca, _, err := finiteLoss(cfg, a.w, ws, b.bag[0].text, holdLo, holdHi)
	if err != nil {
		t.Fatal(err)
	}
	cb, _, err := finiteLoss(cfg, b.w, ws, a.bag[0].text, holdLo, holdHi)
	if err != nil {
		t.Fatal(err)
	}
	a.bag[0].hold = ca
	b.bag[0].hold = cb
	swapped, err := exchange(&a, &b, cfg, ws, g, 0, 1, holdLo, holdLo, holdHi, 1e-2)
	if err != nil {
		t.Fatal(err)
	}
	if swapped || a.bag[0].id != 0 || b.bag[0].id != 2 {
		t.Fatalf("equal sum swapped ids %d %d", a.bag[0].id, b.bag[0].id)
	}

	// One side gets worse, and the other's gain is smaller than that regression.
	a = mk(0, 20, []chunk{{id: 0, text: pat('a'), hold: ca - 1}, {id: 1, text: pat('b'), hold: -1e9}})
	b = mk(1, 21, []chunk{{id: 2, text: pat('c'), hold: cb - 1}, {id: 3, text: pat('d'), hold: -1e9}})
	swapped, err = exchange(&a, &b, cfg, ws, g, 0, 1, holdLo, holdLo, holdHi, 1e-2)
	if err != nil {
		t.Fatal(err)
	}
	if swapped {
		t.Fatal("a higher sum swapped")
	}

	// The sum falls even though one model scores the received window worse.
	a = mk(0, 22, []chunk{{id: 0, text: pat('a'), hold: ca + 5}, {id: 1, text: pat('b'), hold: -1e9}})
	b = mk(1, 23, []chunk{{id: 2, text: pat('c'), hold: cb - 1}, {id: 3, text: pat('d'), hold: -1e9}})
	swapped, err = exchange(&a, &b, cfg, ws, g, 1, 1, holdLo, holdLo, holdHi, 1e-2)
	if err != nil {
		t.Fatal(err)
	}
	if !swapped || a.bag[0].id != 2 || b.bag[0].id != 0 {
		t.Fatalf("sum-improving pair did not swap ids %d %d", a.bag[0].id, b.bag[0].id)
	}
	if a.bag[1].id != 1 || b.bag[1].id != 3 {
		t.Fatal("a window that was not offered moved")
	}
	if a.trades != 1 || b.trades != 1 {
		t.Fatalf("trade counts %d %d", a.trades, b.trades)
	}
	if math.IsNaN(a.own) || math.IsNaN(b.own) || a.own > 20 || b.own > 20 {
		t.Fatalf("post-trade loss %g %g", a.own, b.own)
	}
	if _, err := checkChunks([]node{a, b}); err != nil {
		t.Fatal(err)
	}
}

func TestAllocateNodes(t *testing.T) {
	got, err := allocateNodes([]int{10, 30, 0}, []int{2, 100, 5}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 2 || got[1] != 8 || got[2] != 0 {
		t.Fatalf("allocation %v", got)
	}
	if _, err := allocateNodes([]int{4}, []int{1}, 4); err == nil {
		t.Fatal("expected capacity to reject the request")
	}
}

func TestAssignWorks(t *testing.T) {
	data, err := os.ReadFile("pg100.txt")
	if err != nil {
		t.Fatal(err)
	}
	works, err := parseWorks(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(works) != 44 {
		t.Fatalf("got %d works", len(works))
	}
	if works[0].name != "THE SONNETS" {
		t.Fatalf("first work %q", works[0].name)
	}
	for i, w := range works {
		if w.end <= w.start {
			t.Fatalf("%s empty span", w.name)
		}
		if i > 0 && w.start != works[i-1].end {
			t.Fatalf("%s does not start where %s ends", w.name, works[i-1].name)
		}
		head := strings.TrimRight(string(data[w.start:min(w.start+len(w.name), w.end)]), "\r")
		if !strings.HasPrefix(head, w.name) {
			t.Fatalf("%s body does not start at the title", w.name)
		}
	}
	if sceneStart(data, works[0].start, works[0].end) != works[0].start {
		t.Fatal("a poem moved off its title")
	}
	var play work
	for _, w := range works {
		if strings.Contains(w.name, "ALL") && strings.Contains(w.name, "WELL") {
			play = w
			break
		}
	}
	if play.name == "" || sceneStart(data, play.start, play.end) <= play.start {
		t.Fatal("a play did not skip its contents page")
	}

	const context, perNode, nodesN = 1024, 4, 1024
	nodes, err := assignNodes(data, works, nodesN, context, perNode, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes) != nodesN {
		t.Fatalf("assigned %d", len(nodes))
	}
	if _, err := checkChunks(nodes); err != nil {
		t.Fatal(err)
	}
	seenOff := map[int]bool{}
	for _, n := range nodes {
		if len(n.bag) != perNode || n.origin == "" {
			t.Fatalf("node %d bag %d origin %q", n.id, len(n.bag), n.origin)
		}
		for j, c := range n.bag {
			if c.work != n.origin || len(c.text) != context {
				t.Fatalf("node %d window %d work %q len %d", n.id, j, c.work, len(c.text))
			}
			if c.off < 0 || c.off+context > len(data) {
				t.Fatalf("window %d offset %d", c.id, c.off)
			}
			if seenOff[c.off] {
				t.Fatalf("duplicate offset %d", c.off)
			}
			seenOff[c.off] = true
			if j > 0 && c.off < n.bag[j-1].off+context {
				t.Fatalf("overlapping windows at node %d", n.id)
			}
		}
	}
	if _, err := assignNodes(data, works, 100000, context, perNode, 1); err == nil {
		t.Fatal("expected too many nodes to fail")
	}
	if _, err := parseWorks([]byte("no contents here")); err == nil {
		t.Fatal("expected a file without a contents list to fail")
	}
}
