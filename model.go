// Copyright 2026 The trader Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand/v2"
)

// Config is one pre-norm causal transformer layer.
// A window of Context bytes is scored as next-byte cross-entropy.
// Position i reads bytes 0..i and is trained to predict byte i+1.
// The last byte is a target only. Attention is causal, so a byte never
// sees a later byte.
type Config struct {
	Vocab   int
	Context int
	D       int
	Heads   int
	DFF     int
}

func simConfig(context int) Config {
	return Config{Vocab: 256, Context: context, D: 16, Heads: 2, DFF: 32}
}

func (c Config) validate() error {
	if c.Vocab < 2 || c.Context < 2 || c.D < 1 || c.Heads < 1 || c.DFF < 1 || c.D%c.Heads != 0 {
		return errConfig
	}
	return nil
}

func (c Config) dh() int { return c.D / c.Heads }

func (c Config) preds() int { return c.Context - 1 }

// tensors is one node's parameters.
type tensors struct {
	Emb, Pos       []float32
	Ln1W, Ln1B     []float32
	Wq, Wk, Wv, Wo []float32
	Bq, Bk, Bv, Bo []float32
	Ln2W, Ln2B     []float32
	W1, B1         []float32
	W2, B2         []float32
	LnFW, LnFB     []float32
	Wh, Bh         []float32
}

func (t *tensors) list() [][]float32 {
	return [][]float32{
		t.Emb, t.Pos,
		t.Ln1W, t.Ln1B,
		t.Wq, t.Wk, t.Wv, t.Wo,
		t.Bq, t.Bk, t.Bv, t.Bo,
		t.Ln2W, t.Ln2B,
		t.W1, t.B1, t.W2, t.B2,
		t.LnFW, t.LnFB,
		t.Wh, t.Bh,
	}
}

func (t *tensors) zero() {
	for _, p := range t.list() {
		clear(p)
	}
}

func newTensors(cfg Config) *tensors {
	if err := cfg.validate(); err != nil {
		panic(err)
	}
	a := func(n int) []float32 { return make([]float32, n) }
	d, v, t, dff := cfg.D, cfg.Vocab, cfg.Context, cfg.DFF
	return &tensors{
		Emb: a(v * d), Pos: a(t * d),
		Ln1W: a(d), Ln1B: a(d),
		Wq: a(d * d), Wk: a(d * d), Wv: a(d * d), Wo: a(d * d),
		Bq: a(d), Bk: a(d), Bv: a(d), Bo: a(d),
		Ln2W: a(d), Ln2B: a(d),
		W1: a(d * dff), B1: a(dff),
		W2: a(dff * d), B2: a(d),
		LnFW: a(d), LnFB: a(d),
		Wh: a(d * v), Bh: a(v),
	}
}

func paramCount(w *tensors) int {
	n := 0
	for _, p := range w.list() {
		n += len(p)
	}
	return n
}

func initNormal(r *rand.Rand, w []float32, std float64) {
	for i := range w {
		w[i] = float32(r.NormFloat64() * std)
	}
}

func initTensors(w *tensors, r *rand.Rand) {
	const std = 0.02
	initNormal(r, w.Emb, std)
	initNormal(r, w.Pos, std)
	initNormal(r, w.Wq, std)
	initNormal(r, w.Wk, std)
	initNormal(r, w.Wv, std)
	initNormal(r, w.Wo, std)
	initNormal(r, w.W1, std)
	initNormal(r, w.W2, std)
	initNormal(r, w.Wh, std)
	for i := range w.Ln1W {
		w.Ln1W[i] = 1
		w.Ln2W[i] = 1
		w.LnFW[i] = 1
	}
}

type workspace struct {
	T, D, H, Dh, DFF, V int
	h, ln1              []float32
	ln1Mean, ln1Rstd    []float32
	q, k, v             []float32
	scores, attn        []float32
	ctx, ao, h2         []float32
	ln2                 []float32
	ln2Mean, ln2Rstd    []float32
	pre, hid, fo, h3    []float32
	lnf                 []float32
	lnfMean, lnfRstd    []float32
	logits, glogits     []float32

	qh, kh, vh, ctxh []float32
	gctxh, dqh       []float32
	kt, vt, gkt, gvt []float32
	ds               []float32

	gh, gln1         []float32
	gq, gk, gv, gctx []float32
	gh2              []float32
	gln2, gpre, ghid []float32
	gh3, glnf        []float32
	dx               []float32
}

func newWorkspace(cfg Config) *workspace {
	if err := cfg.validate(); err != nil {
		panic(err)
	}
	T, d, h, dff, v := cfg.Context, cfg.D, cfg.Heads, cfg.DFF, cfg.Vocab
	dh := d / h
	f := func(n int) []float32 { return make([]float32, n) }
	td := T * d
	return &workspace{
		T: T, D: d, H: h, Dh: dh, DFF: dff, V: v,
		h: f(td), ln1: f(td), ln1Mean: f(T), ln1Rstd: f(T),
		q: f(td), k: f(td), v: f(td),
		scores: f(T * T), attn: f(h * T * T),
		ctx: f(td), ao: f(td), h2: f(td),
		ln2: f(td), ln2Mean: f(T), ln2Rstd: f(T),
		pre: f(T * dff), hid: f(T * dff), fo: f(td), h3: f(td),
		lnf: f(td), lnfMean: f(T), lnfRstd: f(T),
		logits: f(T * v), glogits: f(T * v),
		qh: f(T * dh), kh: f(T * dh), vh: f(T * dh), ctxh: f(T * dh),
		gctxh: f(T * dh), dqh: f(T * dh),
		kt: f(dh * T), vt: f(dh * T), gkt: f(dh * T), gvt: f(dh * T),
		ds: f(T * T),
		gh: f(td), gln1: f(td),
		gq: f(td), gk: f(td), gv: f(td), gctx: f(td),
		gh2:  f(td),
		gln2: f(td), gpre: f(T * dff), ghid: f(T * dff),
		gh3: f(td), glnf: f(td),
		dx: f(td),
	}
}

func readHead(src []float32, T, D, h, Dh int, dst []float32) {
	for i := 0; i < T; i++ {
		copy(dst[i*Dh:(i+1)*Dh], src[i*D+h*Dh:i*D+h*Dh+Dh])
	}
}

func writeHead(dst []float32, T, D, h, Dh int, src []float32) {
	for i := 0; i < T; i++ {
		copy(dst[i*D+h*Dh:i*D+h*Dh+Dh], src[i*Dh:(i+1)*Dh])
	}
}

func addHead(dst []float32, T, D, h, Dh int, src []float32) {
	for i := 0; i < T; i++ {
		di := dst[i*D+h*Dh : i*D+h*Dh+Dh]
		si := src[i*Dh : (i+1)*Dh]
		for d := 0; d < Dh; d++ {
			di[d] += si[d]
		}
	}
}

func transposeTD(src []float32, T, Dh int, dst []float32) {
	for d := 0; d < Dh; d++ {
		col := dst[d*T : (d+1)*T]
		for i := 0; i < T; i++ {
			col[i] = src[i*Dh+d]
		}
	}
}

func addTransposed(dst []float32, T, D, h, Dh int, src []float32) {
	for d := 0; d < Dh; d++ {
		col := src[d*T : (d+1)*T]
		for i := 0; i < T; i++ {
			dst[i*D+h*Dh+d] += col[i]
		}
	}
}

// qkRows writes scores[i, j] = scale * dot(q[i], k[:, j]).
// q is [T, Dh]. kt is [Dh, T].
func qkRows(q, kt []float32, T, Dh int, scale float32, scores []float32) {
	for i := 0; i < T; i++ {
		row := scores[i*T : (i+1)*T]
		clear(row)
		qi := q[i*Dh : (i+1)*Dh]
		for d := 0; d < Dh; d++ {
			axpy(row, kt[d*T:(d+1)*T], qi[d]*scale)
		}
	}
}

// avRows writes out[i] = sum_j attn[i, j] * v[:, j].
func avRows(attn, vt []float32, T, Dh int, out []float32) {
	for i := 0; i < T; i++ {
		ai := attn[i*T : (i+1)*T]
		oi := out[i*Dh : (i+1)*Dh]
		for d := 0; d < Dh; d++ {
			oi[d] = dot(ai, vt[d*T:(d+1)*T])
		}
	}
}

func causalSoftmax(scores, attn []float32, T int) {
	for i := 0; i < T; i++ {
		base := i * T
		max := scores[base]
		for j := 1; j <= i; j++ {
			if scores[base+j] > max {
				max = scores[base+j]
			}
		}
		var sum float64
		for j := 0; j <= i; j++ {
			e := math.Exp(float64(scores[base+j] - max))
			attn[base+j] = float32(e)
			sum += e
		}
		inv := float32(1 / sum)
		for j := 0; j <= i; j++ {
			attn[base+j] *= inv
		}
		for j := i + 1; j < T; j++ {
			attn[base+j] = 0
		}
	}
}

func attnForward(ws *workspace, scale float32) {
	T, H, Dh, D := ws.T, ws.H, ws.Dh, ws.D
	for h := 0; h < H; h++ {
		readHead(ws.q, T, D, h, Dh, ws.qh)
		readHead(ws.k, T, D, h, Dh, ws.kh)
		readHead(ws.v, T, D, h, Dh, ws.vh)
		transposeTD(ws.kh, T, Dh, ws.kt)
		qkRows(ws.qh, ws.kt, T, Dh, scale, ws.scores)
		ah := ws.attn[h*T*T : (h+1)*T*T]
		causalSoftmax(ws.scores, ah, T)
		transposeTD(ws.vh, T, Dh, ws.vt)
		avRows(ah, ws.vt, T, Dh, ws.ctxh)
		writeHead(ws.ctx, T, D, h, Dh, ws.ctxh)
	}
}

func attnBackward(ws *workspace, scale float32) {
	T, H, Dh, D := ws.T, ws.H, ws.Dh, ws.D
	clear(ws.gq)
	clear(ws.gk)
	clear(ws.gv)
	for h := 0; h < H; h++ {
		readHead(ws.q, T, D, h, Dh, ws.qh)
		readHead(ws.k, T, D, h, Dh, ws.kh)
		readHead(ws.v, T, D, h, Dh, ws.vh)
		readHead(ws.gctx, T, D, h, Dh, ws.gctxh)
		transposeTD(ws.kh, T, Dh, ws.kt)
		transposeTD(ws.vh, T, Dh, ws.vt)
		// dA[i, j] = dot(gctx[i], v[j]).
		qkRows(ws.gctxh, ws.vt, T, Dh, 1, ws.scores)
		ah := ws.attn[h*T*T : (h+1)*T*T]
		clear(ws.gvt)
		for i := 0; i < T; i++ {
			gi := ws.gctxh[i*Dh : (i+1)*Dh]
			ai := ah[i*T : (i+1)*T]
			for d := 0; d < Dh; d++ {
				axpy(ws.gvt[d*T:(d+1)*T], ai, gi[d])
			}
		}
		for i := 0; i < T; i++ {
			base := i * T
			var dotA float32
			for j := 0; j <= i; j++ {
				dotA += ws.scores[base+j] * ah[base+j]
			}
			for j := 0; j <= i; j++ {
				ws.ds[base+j] = ah[base+j] * (ws.scores[base+j] - dotA) * scale
			}
			for j := i + 1; j < T; j++ {
				ws.ds[base+j] = 0
			}
		}
		for i := 0; i < T; i++ {
			di := ws.ds[i*T : (i+1)*T]
			oi := ws.dqh[i*Dh : (i+1)*Dh]
			for d := 0; d < Dh; d++ {
				oi[d] = dot(di, ws.kt[d*T:(d+1)*T])
			}
		}
		clear(ws.gkt)
		for i := 0; i < T; i++ {
			qi := ws.qh[i*Dh : (i+1)*Dh]
			di := ws.ds[i*T : (i+1)*T]
			for d := 0; d < Dh; d++ {
				axpy(ws.gkt[d*T:(d+1)*T], di, qi[d])
			}
		}
		addHead(ws.gq, T, D, h, Dh, ws.dqh)
		addTransposed(ws.gk, T, D, h, Dh, ws.gkt)
		addTransposed(ws.gv, T, D, h, Dh, ws.gvt)
	}
}

// rowCE is the cross-entropy of one next-byte distribution.
// Ties break toward the smaller index. A nil dlogits skips the gradient.
func rowCE(logits []float32, target int, dlogits []float32) (loss float64, correct int) {
	V := len(logits)
	if target < 0 || target >= V {
		panic("target")
	}
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v - max))
	}
	loss = -(float64(logits[target]-max) - math.Log(sum))
	best, arg := logits[0], 0
	inv := 1 / sum
	for i, v := range logits {
		if v > best {
			best, arg = v, i
		}
		if dlogits != nil {
			g := float32(math.Exp(float64(v-max)) * inv)
			if i == target {
				g -= 1
			}
			dlogits[i] = g
		}
	}
	if arg == target {
		correct = 1
	}
	return loss, correct
}

// forwardBackward runs one window. text must be Context bytes.
// Loss is the mean next-byte cross-entropy over the first Context-1 positions.
// A nil g skips the backward pass. Gradients are the derivative of that mean.
func forwardBackward(cfg Config, w, g *tensors, ws *workspace, text []byte) (loss float64, correct int) {
	T, D, V := cfg.Context, cfg.D, cfg.Vocab
	if ws.T != T || ws.D != D || ws.V != V || len(text) != T {
		panic("batch shape")
	}
	for i := 0; i < T; i++ {
		id := int(text[i])
		if id < 0 || id >= V {
			panic("byte out of vocab")
		}
		hi := ws.h[i*D : (i+1)*D]
		ei := w.Emb[id*D : (id+1)*D]
		pi := w.Pos[i*D : (i+1)*D]
		for d := 0; d < D; d++ {
			hi[d] = ei[d] + pi[d]
		}
	}
	Dh, DFF := ws.Dh, ws.DFF
	scale := 1 / sqrt32(float32(Dh))

	layernorm(ws.h, T, D, w.Ln1W, w.Ln1B, ws.ln1, ws.ln1Mean, ws.ln1Rstd)
	mul(ws.ln1, T, D, w.Wq, D, ws.q)
	addBias(ws.q, T, D, w.Bq)
	mul(ws.ln1, T, D, w.Wk, D, ws.k)
	addBias(ws.k, T, D, w.Bk)
	mul(ws.ln1, T, D, w.Wv, D, ws.v)
	addBias(ws.v, T, D, w.Bv)
	attnForward(ws, scale)
	mul(ws.ctx, T, D, w.Wo, D, ws.ao)
	addBias(ws.ao, T, D, w.Bo)
	for i := 0; i < T*D; i++ {
		ws.h2[i] = ws.h[i] + ws.ao[i]
	}
	layernorm(ws.h2, T, D, w.Ln2W, w.Ln2B, ws.ln2, ws.ln2Mean, ws.ln2Rstd)
	mul(ws.ln2, T, D, w.W1, DFF, ws.pre)
	addBias(ws.pre, T, DFF, w.B1)
	applyGelu(ws.pre, ws.hid)
	mul(ws.hid, T, DFF, w.W2, D, ws.fo)
	addBias(ws.fo, T, D, w.B2)
	for i := 0; i < T*D; i++ {
		ws.h3[i] = ws.h2[i] + ws.fo[i]
	}
	layernorm(ws.h3, T, D, w.LnFW, w.LnFB, ws.lnf, ws.lnfMean, ws.lnfRstd)
	mul(ws.lnf, T, D, w.Wh, V, ws.logits)
	addBias(ws.logits, T, V, w.Bh)

	npred := T - 1
	var dlogits []float32
	if g != nil {
		clear(ws.glogits)
		dlogits = ws.glogits
	}
	for i := 0; i < npred; i++ {
		var grow []float32
		if dlogits != nil {
			grow = dlogits[i*V : (i+1)*V]
		}
		li, c := rowCE(ws.logits[i*V:(i+1)*V], int(text[i+1]), grow)
		loss += li
		correct += c
	}
	loss /= float64(npred)
	if g == nil {
		return loss, correct
	}
	inv := float32(1 / float64(npred))
	for i := 0; i < npred*V; i++ {
		ws.glogits[i] *= inv
	}
	g.zero()

	mulAT(ws.lnf, T, D, ws.glogits, V, g.Wh)
	sumRows(ws.glogits, T, V, g.Bh)
	mulBT(ws.glogits, T, V, w.Wh, D, ws.glnf)
	layernormBackward(ws.glnf, ws.h3, w.LnFW, ws.lnfMean, ws.lnfRstd, T, D, ws.gh3, g.LnFW, g.LnFB)

	mulAT(ws.hid, T, DFF, ws.gh3, D, g.W2)
	sumRows(ws.gh3, T, D, g.B2)
	mulBT(ws.gh3, T, D, w.W2, DFF, ws.ghid)
	applyDGelu(ws.pre, ws.ghid, ws.gpre)
	mulAT(ws.ln2, T, D, ws.gpre, DFF, g.W1)
	sumRows(ws.gpre, T, DFF, g.B1)
	mulBT(ws.gpre, T, DFF, w.W1, D, ws.gln2)
	layernormBackward(ws.gln2, ws.h2, w.Ln2W, ws.ln2Mean, ws.ln2Rstd, T, D, ws.dx, g.Ln2W, g.Ln2B)
	for i := 0; i < T*D; i++ {
		ws.gh2[i] = ws.gh3[i] + ws.dx[i]
	}

	mulAT(ws.ctx, T, D, ws.gh2, D, g.Wo)
	sumRows(ws.gh2, T, D, g.Bo)
	mulBT(ws.gh2, T, D, w.Wo, D, ws.gctx)
	attnBackward(ws, scale)

	mulAT(ws.ln1, T, D, ws.gq, D, g.Wq)
	sumRows(ws.gq, T, D, g.Bq)
	mulBT(ws.gq, T, D, w.Wq, D, ws.gln1)
	mulAT(ws.ln1, T, D, ws.gk, D, g.Wk)
	sumRows(ws.gk, T, D, g.Bk)
	mulBT(ws.gk, T, D, w.Wk, D, ws.dx)
	for i := 0; i < T*D; i++ {
		ws.gln1[i] += ws.dx[i]
	}
	mulAT(ws.ln1, T, D, ws.gv, D, g.Wv)
	sumRows(ws.gv, T, D, g.Bv)
	mulBT(ws.gv, T, D, w.Wv, D, ws.dx)
	for i := 0; i < T*D; i++ {
		ws.gln1[i] += ws.dx[i]
	}
	layernormBackward(ws.gln1, ws.h, w.Ln1W, ws.ln1Mean, ws.ln1Rstd, T, D, ws.dx, g.Ln1W, g.Ln1B)
	for i := 0; i < T*D; i++ {
		ws.gh[i] = ws.gh2[i] + ws.dx[i]
	}
	for i := 0; i < T; i++ {
		id := int(text[i])
		gh := ws.gh[i*D : (i+1)*D]
		ge := g.Emb[id*D : (id+1)*D]
		gp := g.Pos[i*D : (i+1)*D]
		for d := 0; d < D; d++ {
			ge[d] += gh[d]
			gp[d] += gh[d]
		}
	}
	return loss, correct
}

type adam struct {
	m, v        [][]float32
	t           int
	b1, b2, eps float32
}

func newAdam(w *tensors) *adam {
	a := &adam{b1: 0.9, b2: 0.999, eps: 1e-8}
	for _, p := range w.list() {
		a.m = append(a.m, make([]float32, len(p)))
		a.v = append(a.v, make([]float32, len(p)))
	}
	return a
}

func (a *adam) step(w, g *tensors, lr float32) {
	a.t++
	c1 := 1 - float32(math.Pow(float64(a.b1), float64(a.t)))
	c2 := 1 - float32(math.Pow(float64(a.b2), float64(a.t)))
	ws, gs := w.list(), g.list()
	for pi := range ws {
		pw, pg, pm, pv := ws[pi], gs[pi], a.m[pi], a.v[pi]
		for i := range pw {
			gi := pg[i]
			pm[i] = a.b1*pm[i] + (1-a.b1)*gi
			pv[i] = a.b2*pv[i] + (1-a.b2)*gi*gi
			pw[i] -= lr * (pm[i] / c1) / (sqrt32(pv[i]/c2) + a.eps)
		}
	}
}

func gradNorm(g *tensors) float64 {
	var ss float64
	for _, p := range g.list() {
		for _, v := range p {
			ss += float64(v) * float64(v)
		}
	}
	return math.Sqrt(ss)
}

func clipGrads(g *tensors, maxNorm float32) float64 {
	norm := gradNorm(g)
	if norm > float64(maxNorm) && norm > 0 && !math.IsNaN(norm) && !math.IsInf(norm, 0) {
		s := float32(float64(maxNorm) / norm)
		for _, p := range g.list() {
			for i := range p {
				p[i] *= s
			}
		}
	}
	return norm
}
