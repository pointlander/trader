// Copyright 2026 The trader Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand/v2"
	"os"
	"testing"
)

func TestFiniteDifference(t *testing.T) {
	cfg := Config{Vocab: 16, Context: 6, D: 8, Heads: 2, DFF: 16}
	w := newTensors(cfg)
	initTensors(w, rand.New(rand.NewPCG(3, 1)))
	ws := newWorkspace(cfg)
	g := newTensors(cfg)
	text := make([]byte, cfg.Context)
	for i := range text {
		text[i] = byte((i*3 + 1) % cfg.Vocab)
	}
	forwardBackward(cfg, w, g, ws, text)

	checks := []struct {
		name string
		p, g []float32
		idx  []int
	}{
		{"Emb", w.Emb, g.Emb, []int{0, cfg.D, 2 * cfg.D}},
		{"Pos", w.Pos, g.Pos, []int{0, cfg.D, cfg.D * (cfg.Context - 1)}},
		{"Ln1W", w.Ln1W, g.Ln1W, []int{0, cfg.D / 2, cfg.D - 1}},
		{"Wq", w.Wq, g.Wq, []int{0, len(w.Wq) / 2, len(w.Wq) - 1}},
		{"Wo", w.Wo, g.Wo, []int{1, len(w.Wo) / 3, len(w.Wo) - 2}},
		{"W1", w.W1, g.W1, []int{0, len(w.W1) / 2, len(w.W1) - 1}},
		{"W2", w.W2, g.W2, []int{0, 7, len(w.W2) - 1}},
		{"Wh", w.Wh, g.Wh, []int{0, cfg.D, len(w.Wh) - 1}},
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
	lp, _ := forwardBackward(cfg, w, nil, ws, text)
	p[idx] -= 2 * eps
	lm, _ := forwardBackward(cfg, w, nil, ws, text)
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
	cfg := Config{Vocab: 16, Context: 6, D: 8, Heads: 2, DFF: 16}
	w := newTensors(cfg)
	initTensors(w, rand.New(rand.NewPCG(4, 2)))
	ws := newWorkspace(cfg)
	text := []byte{1, 2, 3, 4, 5, 6}
	forwardBackward(cfg, w, nil, ws, text)
	first := append([]float32(nil), ws.logits[:cfg.Vocab*(cfg.Context-1)]...)
	text[len(text)-1] = 9
	forwardBackward(cfg, w, nil, ws, text)
	for i, v := range first {
		if ws.logits[i] != v {
			t.Fatalf("future byte changed logit %d", i)
		}
	}
}

func TestLearnsBigram(t *testing.T) {
	cfg := simConfig(32)
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
	before, _, err := finiteLoss(cfg, w, ws, text)
	if err != nil {
		t.Fatal(err)
	}
	for s := 0; s < 40; s++ {
		if _, err := learnStep(cfg, w, g, opt, ws, text, 1e-2); err != nil {
			t.Fatal(err)
		}
	}
	after, acc, err := finiteLoss(cfg, w, ws, text)
	if err != nil {
		t.Fatal(err)
	}
	if !(after < before-1) {
		t.Fatalf("loss %.3f -> %.3f, acc %.3f", before, after, acc)
	}
}

func TestTradeRule(t *testing.T) {
	cfg := simConfig(16)
	mk := func(id int, text []byte) node {
		w := newTensors(cfg)
		initTensors(w, rand.New(rand.NewPCG(uint64(10+id), 1)))
		return node{id: id, src: id, text: text, w: w, opt: newAdam(w), own: 0}
	}
	aText := bytesOf("abababababababab")
	bText := bytesOf("cdcdcdcdcdcdcdcd")
	ws := newWorkspace(cfg)
	g := newTensors(cfg)

	a := mk(0, aText)
	b := mk(1, bText)
	a.own, b.own = -1, -1
	outcome, _, _, _, _, err := considerPair(&a, &b, cfg, ws, g, 1, 1e-2)
	if err != nil {
		t.Fatal(err)
	}
	if outcome != pairNeither || a.src != 0 || b.src != 1 {
		t.Fatalf("easy own loss traded: outcome=%d src=%d,%d", outcome, a.src, b.src)
	}

	a = mk(0, append([]byte(nil), aText...))
	b = mk(1, append([]byte(nil), bText...))
	a.own, b.own = 1e9, -1
	outcome, _, _, _, _, err = considerPair(&a, &b, cfg, ws, g, 1, 1e-2)
	if err != nil {
		t.Fatal(err)
	}
	if outcome != pairOne || a.src != 0 || b.src != 1 {
		t.Fatalf("one-sided pair swapped: outcome=%d src=%d,%d", outcome, a.src, b.src)
	}

	a = mk(0, append([]byte(nil), aText...))
	b = mk(1, append([]byte(nil), bText...))
	a.own, b.own = 1e9, 1e9
	outcome, _, _, _, _, err = considerPair(&a, &b, cfg, ws, g, 2, 1e-2)
	if err != nil {
		t.Fatal(err)
	}
	if outcome != pairTrade || a.src != 1 || b.src != 0 {
		t.Fatalf("mutual pair did not swap: outcome=%d src=%d,%d", outcome, a.src, b.src)
	}
	if a.trades != 1 || b.trades != 1 {
		t.Fatalf("trade counts %d %d", a.trades, b.trades)
	}
	if !(a.own < 100 && b.own < 100) {
		t.Fatalf("post-trade loss %g %g", a.own, b.own)
	}
	if string(a.text) != string(bText) || string(b.text) != string(aText) {
		t.Fatal("windows did not move")
	}
}

func TestWindowStarts(t *testing.T) {
	data, err := os.ReadFile("pg100.txt")
	if err != nil {
		t.Fatal(err)
	}
	starts, err := windowStarts(len(data), 1024, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if len(starts) != 1024 {
		t.Fatalf("got %d windows", len(starts))
	}
	for i, s := range starts {
		if s < 0 || s+1024 > len(data) {
			t.Fatalf("window %d offset %d", i, s)
		}
		if i > 0 && s < starts[i-1]+1024 {
			t.Fatalf("overlap at %d (%d, %d)", i, starts[i-1], s)
		}
	}
	if _, err := windowStarts(100, 1024, 2); err == nil {
		t.Fatal("expected a short file to fail")
	}
}

func bytesOf(s string) []byte { return []byte(s) }
