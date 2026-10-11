// Copyright 2026 The trader Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand/v2"
)

// Config is a stack of pre-norm causal transformer layers.
// Position i reads bytes 0..i and is trained to predict byte i+1.
// Positions come from rotary embeddings applied to queries and keys.
// The byte embedding is tied to the output projection.
// A loss span scores only targets whose index lies in [tgtLo, tgtHi).
// Bytes at index tgtHi and beyond are not inputs, so a held-out tail
// stays out of the training sequence.
type Config struct {
	Vocab   int
	Context int
	D       int
	Heads   int
	DFF     int
	Layers  int
}

func simConfig(context int) Config {
	return Config{Vocab: 256, Context: context, D: 32, Heads: 4, DFF: 128, Layers: 2}
}

func (c Config) validate() error {
	if c.Vocab < 2 || c.Context < 2 || c.D < 1 || c.Heads < 1 || c.DFF < 1 || c.Layers < 1 || c.D%c.Heads != 0 {
		return errConfig
	}
	if (c.D/c.Heads)%2 != 0 {
		return errConfig
	}
	return nil
}

func (c Config) dh() int { return c.D / c.Heads }

// tensors is one node's parameters. The output projection is Emb.
type tensors struct {
	Emb        []float32
	Bh         []float32
	LnFW, LnFB []float32
	Layers     []layer
}

type layer struct {
	Ln1W, Ln1B     []float32
	Wq, Wk, Wv, Wo []float32
	Bq, Bk, Bv, Bo []float32
	Ln2W, Ln2B     []float32
	W1, B1         []float32
	W2, B2         []float32
}

func (t *tensors) list() [][]float32 {
	out := make([][]float32, 0, 4+len(t.Layers)*16)
	out = append(out, t.Emb, t.Bh, t.LnFW, t.LnFB)
	for i := range t.Layers {
		L := &t.Layers[i]
		out = append(out,
			L.Ln1W, L.Ln1B,
			L.Wq, L.Wk, L.Wv, L.Wo,
			L.Bq, L.Bk, L.Bv, L.Bo,
			L.Ln2W, L.Ln2B,
			L.W1, L.B1, L.W2, L.B2,
		)
	}
	return out
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
	d, v, dff := cfg.D, cfg.Vocab, cfg.DFF
	t := &tensors{
		Emb: a(v * d), Bh: a(v),
		LnFW: a(d), LnFB: a(d),
		Layers: make([]layer, cfg.Layers),
	}
	for i := range t.Layers {
		t.Layers[i] = layer{
			Ln1W: a(d), Ln1B: a(d),
			Wq: a(d * d), Wk: a(d * d), Wv: a(d * d), Wo: a(d * d),
			Bq: a(d), Bk: a(d), Bv: a(d), Bo: a(d),
			Ln2W: a(d), Ln2B: a(d),
			W1: a(d * dff), B1: a(dff),
			W2: a(dff * d), B2: a(d),
		}
	}
	return t
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
	for i := range w.Layers {
		L := &w.Layers[i]
		initNormal(r, L.Wq, std)
		initNormal(r, L.Wk, std)
		initNormal(r, L.Wv, std)
		initNormal(r, L.Wo, std)
		initNormal(r, L.W1, std)
		initNormal(r, L.W2, std)
		for j := range L.Ln1W {
			L.Ln1W[j] = 1
			L.Ln2W[j] = 1
		}
	}
	for i := range w.LnFW {
		w.LnFW[i] = 1
	}
}

type layerAct struct {
	h, ln1           []float32
	ln1Mean, ln1Rstd []float32
	q, k, v          []float32
	attn             []float32
	ctx, ao, h2      []float32
	ln2              []float32
	ln2Mean, ln2Rstd []float32
	pre, hid, fo, h3 []float32
}

type workspace struct {
	maxT                int
	T, D, H, Dh, DFF, V int
	blocks              []layerAct
	lnf                 []float32
	lnfMean, lnfRstd    []float32
	logits, glogits     []float32
	tok                 []float32
	ropeCos, ropeSin    []float32

	scores, ds          []float32
	qh, kh, vh, ctxh    []float32
	gctxh, dqh          []float32
	kt, vt, gkt, gvt    []float32
	gq, gk, gv, gctx    []float32
	gh, gln1, gh2, gln2 []float32
	gpre, ghid, gh3, dx []float32
	glnf, upstream      []float32
}

func newWorkspace(cfg Config) *workspace {
	if err := cfg.validate(); err != nil {
		panic(err)
	}
	T, d, h, dff, v := cfg.Context, cfg.D, cfg.Heads, cfg.DFF, cfg.Vocab
	dh := d / h
	f := func(n int) []float32 { return make([]float32, n) }
	td := T * d
	ws := &workspace{
		maxT: T, T: T, D: d, H: h, Dh: dh, DFF: dff, V: v,
		blocks: make([]layerAct, cfg.Layers),
		lnf:    f(td), lnfMean: f(T), lnfRstd: f(T),
		logits: f(T * v), glogits: f(T * v),
		tok:     f(td),
		ropeCos: f(T * (dh / 2)), ropeSin: f(T * (dh / 2)),
		scores: f(T * T), ds: f(T * T),
		qh: f(T * dh), kh: f(T * dh), vh: f(T * dh), ctxh: f(T * dh),
		gctxh: f(T * dh), dqh: f(T * dh),
		kt: f(dh * T), vt: f(dh * T), gkt: f(dh * T), gvt: f(dh * T),
		gq: f(td), gk: f(td), gv: f(td), gctx: f(td),
		gh: f(td), gln1: f(td), gh2: f(td), gln2: f(td),
		gpre: f(T * dff), ghid: f(T * dff), gh3: f(td), dx: f(td),
		glnf: f(td), upstream: f(td),
	}
	for i := range ws.blocks {
		ws.blocks[i] = layerAct{
			h: f(td), ln1: f(td), ln1Mean: f(T), ln1Rstd: f(T),
			q: f(td), k: f(td), v: f(td),
			attn: f(h * T * T),
			ctx:  f(td), ao: f(td), h2: f(td),
			ln2: f(td), ln2Mean: f(T), ln2Rstd: f(T),
			pre: f(T * dff), hid: f(T * dff), fo: f(td), h3: f(td),
		}
	}
	buildRope(ws.ropeCos, ws.ropeSin, T, dh)
	return ws
}

func buildRope(cos, sin []float32, T, Dh int) {
	half := Dh / 2
	for i := 0; i < half; i++ {
		theta := math.Pow(10000, -float64(2*i)/float64(Dh))
		for t := 0; t < T; t++ {
			ang := float64(t) * theta
			cos[t*half+i] = float32(math.Cos(ang))
			sin[t*half+i] = float32(math.Sin(ang))
		}
	}
}

// applyRope rotates pairs of channels. sign -1 is the inverse rotation.
func applyRope(x []float32, T, D, H, Dh int, cos, sin []float32, sign float32) {
	half := Dh / 2
	for t := 0; t < T; t++ {
		cs := cos[t*half : (t+1)*half]
		sn := sin[t*half : (t+1)*half]
		for h := 0; h < H; h++ {
			base := t*D + h*Dh
			for i := 0; i < half; i++ {
				x0 := x[base+2*i]
				x1 := x[base+2*i+1]
				c := cs[i]
				s := sign * sn[i]
				x[base+2*i] = x0*c - x1*s
				x[base+2*i+1] = x0*s + x1*c
			}
		}
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

func attnForward(ws *workspace, q, k, v, attn, ctx []float32, scale float32) {
	T, H, Dh, D := ws.T, ws.H, ws.Dh, ws.D
	for h := 0; h < H; h++ {
		readHead(q, T, D, h, Dh, ws.qh)
		readHead(k, T, D, h, Dh, ws.kh)
		readHead(v, T, D, h, Dh, ws.vh)
		transposeTD(ws.kh, T, Dh, ws.kt)
		qkRows(ws.qh, ws.kt, T, Dh, scale, ws.scores)
		ah := attn[h*T*T : (h+1)*T*T]
		causalSoftmax(ws.scores, ah, T)
		transposeTD(ws.vh, T, Dh, ws.vt)
		avRows(ah, ws.vt, T, Dh, ws.ctxh)
		writeHead(ctx, T, D, h, Dh, ws.ctxh)
	}
}

func attnBackward(ws *workspace, q, k, v, attn, gctx, gq, gk, gv []float32, scale float32) {
	T, H, Dh, D := ws.T, ws.H, ws.Dh, ws.D
	clear(gq)
	clear(gk)
	clear(gv)
	for h := 0; h < H; h++ {
		readHead(q, T, D, h, Dh, ws.qh)
		readHead(k, T, D, h, Dh, ws.kh)
		readHead(v, T, D, h, Dh, ws.vh)
		readHead(gctx, T, D, h, Dh, ws.gctxh)
		transposeTD(ws.kh, T, Dh, ws.kt)
		transposeTD(ws.vh, T, Dh, ws.vt)
		qkRows(ws.gctxh, ws.vt, T, Dh, 1, ws.scores)
		ah := attn[h*T*T : (h+1)*T*T]
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
		addHead(gq, T, D, h, Dh, ws.dqh)
		addTransposed(gk, T, D, h, Dh, ws.gkt)
		addTransposed(gv, T, D, h, Dh, ws.gvt)
	}
}

func layerForward(st *layerAct, L *layer, ws *workspace, x []float32, scale float32) {
	T, D, DFF := ws.T, ws.D, ws.DFF
	copy(st.h[:T*D], x[:T*D])
	layernorm(st.h, T, D, L.Ln1W, L.Ln1B, st.ln1, st.ln1Mean, st.ln1Rstd)
	mul(st.ln1, T, D, L.Wq, D, st.q)
	addBias(st.q, T, D, L.Bq)
	mul(st.ln1, T, D, L.Wk, D, st.k)
	addBias(st.k, T, D, L.Bk)
	mul(st.ln1, T, D, L.Wv, D, st.v)
	addBias(st.v, T, D, L.Bv)
	applyRope(st.q, T, D, ws.H, ws.Dh, ws.ropeCos, ws.ropeSin, 1)
	applyRope(st.k, T, D, ws.H, ws.Dh, ws.ropeCos, ws.ropeSin, 1)
	attnForward(ws, st.q, st.k, st.v, st.attn, st.ctx, scale)
	mul(st.ctx, T, D, L.Wo, D, st.ao)
	addBias(st.ao, T, D, L.Bo)
	for i := 0; i < T*D; i++ {
		st.h2[i] = st.h[i] + st.ao[i]
	}
	layernorm(st.h2, T, D, L.Ln2W, L.Ln2B, st.ln2, st.ln2Mean, st.ln2Rstd)
	mul(st.ln2, T, D, L.W1, DFF, st.pre)
	addBias(st.pre, T, DFF, L.B1)
	applyGelu(st.pre[:T*DFF], st.hid[:T*DFF])
	mul(st.hid, T, DFF, L.W2, D, st.fo)
	addBias(st.fo, T, D, L.B2)
	for i := 0; i < T*D; i++ {
		st.h3[i] = st.h2[i] + st.fo[i]
	}
}

func layerBackward(st *layerAct, L, gL *layer, ws *workspace, gh3 []float32, scale float32) []float32 {
	T, D, DFF := ws.T, ws.D, ws.DFF
	copy(ws.upstream[:T*D], gh3[:T*D])

	mulAT(st.hid, T, DFF, ws.upstream, D, gL.W2)
	sumRows(ws.upstream, T, D, gL.B2)
	mulBT(ws.upstream, T, D, L.W2, DFF, ws.ghid)
	applyDGelu(st.pre[:T*DFF], ws.ghid[:T*DFF], ws.gpre[:T*DFF])
	mulAT(st.ln2, T, D, ws.gpre, DFF, gL.W1)
	sumRows(ws.gpre, T, DFF, gL.B1)
	mulBT(ws.gpre, T, DFF, L.W1, D, ws.gln2)
	layernormBackward(ws.gln2, st.h2, L.Ln2W, st.ln2Mean, st.ln2Rstd, T, D, ws.dx, gL.Ln2W, gL.Ln2B)
	for i := 0; i < T*D; i++ {
		ws.gh2[i] = ws.upstream[i] + ws.dx[i]
	}

	mulAT(st.ctx, T, D, ws.gh2, D, gL.Wo)
	sumRows(ws.gh2, T, D, gL.Bo)
	mulBT(ws.gh2, T, D, L.Wo, D, ws.gctx)
	attnBackward(ws, st.q, st.k, st.v, st.attn, ws.gctx, ws.gq, ws.gk, ws.gv, scale)
	applyRope(ws.gq, T, D, ws.H, ws.Dh, ws.ropeCos, ws.ropeSin, -1)
	applyRope(ws.gk, T, D, ws.H, ws.Dh, ws.ropeCos, ws.ropeSin, -1)

	mulAT(st.ln1, T, D, ws.gq, D, gL.Wq)
	sumRows(ws.gq, T, D, gL.Bq)
	mulBT(ws.gq, T, D, L.Wq, D, ws.gln1)
	mulAT(st.ln1, T, D, ws.gk, D, gL.Wk)
	sumRows(ws.gk, T, D, gL.Bk)
	mulBT(ws.gk, T, D, L.Wk, D, ws.dx)
	for i := 0; i < T*D; i++ {
		ws.gln1[i] += ws.dx[i]
	}
	mulAT(st.ln1, T, D, ws.gv, D, gL.Wv)
	sumRows(ws.gv, T, D, gL.Bv)
	mulBT(ws.gv, T, D, L.Wv, D, ws.dx)
	for i := 0; i < T*D; i++ {
		ws.gln1[i] += ws.dx[i]
	}
	layernormBackward(ws.gln1, st.h, L.Ln1W, st.ln1Mean, st.ln1Rstd, T, D, ws.dx, gL.Ln1W, gL.Ln1B)
	for i := 0; i < T*D; i++ {
		ws.gh[i] = ws.gh2[i] + ws.dx[i]
	}
	return ws.gh
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

// lossSpan clamps a target range onto [1, n). n is the text length.
func lossSpan(n, tgtLo, tgtHi int) (lo, hi int) {
	lo, hi = tgtLo, tgtHi
	if lo < 1 {
		lo = 1
	}
	if hi > n {
		hi = n
	}
	return lo, hi
}

// forwardBackward runs one causal sequence.
// The active length is the end of the loss span: bytes at that index and
// beyond are not inputs. Training on a prefix therefore cannot attend to
// the held-out tail. Scoring the tail passes a span that ends at the last
// byte, so the prefix remains context. Loss is the mean cross-entropy of
// targets in [tgtLo, tgtHi). A nil g skips the backward pass.
func forwardBackward(cfg Config, w, g *tensors, ws *workspace, text []byte, tgtLo, tgtHi int) (loss float64, correct int) {
	D, V := cfg.D, cfg.Vocab
	if ws.maxT < 2 || ws.D != D || ws.V != V || ws.H != cfg.Heads || len(ws.blocks) != len(w.Layers) || len(text) < 2 || len(text) > ws.maxT {
		panic("batch shape")
	}
	tgtLo, tgtHi = lossSpan(len(text), tgtLo, tgtHi)
	if tgtHi <= tgtLo {
		panic("loss span")
	}
	// Positions at and after tgtHi do not predict a scored byte, and a
	// causal model cannot need them as inputs to earlier positions.
	ws.T = tgtHi
	T := ws.T
	text = text[:T]
	for i := 0; i < T; i++ {
		id := int(text[i])
		if id < 0 || id >= V {
			panic("byte out of vocab")
		}
		hi := ws.tok[i*D : (i+1)*D]
		ei := w.Emb[id*D : (id+1)*D]
		copy(hi, ei)
	}
	scale := 1 / sqrt32(float32(ws.Dh))
	x := ws.tok
	for i := range w.Layers {
		layerForward(&ws.blocks[i], &w.Layers[i], ws, x, scale)
		x = ws.blocks[i].h3
	}
	layernorm(x, T, D, w.LnFW, w.LnFB, ws.lnf, ws.lnfMean, ws.lnfRstd)
	mulBT(ws.lnf, T, D, w.Emb, V, ws.logits)
	addBias(ws.logits, T, V, w.Bh)

	npred := tgtHi - tgtLo
	var dlogits []float32
	if g != nil {
		clear(ws.glogits)
		dlogits = ws.glogits
	}
	for tgt := tgtLo; tgt < tgtHi; tgt++ {
		i := tgt - 1
		var grow []float32
		if dlogits != nil {
			grow = dlogits[i*V : (i+1)*V]
		}
		li, c := rowCE(ws.logits[i*V:(i+1)*V], int(text[tgt]), grow)
		loss += li
		correct += c
	}
	loss /= float64(npred)
	if g == nil {
		return loss, correct
	}
	inv := float32(1 / float64(npred))
	for tgt := tgtLo; tgt < tgtHi; tgt++ {
		i := tgt - 1
		row := ws.glogits[i*V : (i+1)*V]
		for j := range row {
			row[j] *= inv
		}
	}
	g.zero()

	// Tied output: dEmb = glogits^T @ lnf, dlnf = glogits @ Emb.
	mulAT(ws.glogits, T, V, ws.lnf, D, g.Emb)
	sumRows(ws.glogits, T, V, g.Bh)
	mul(ws.glogits, T, V, w.Emb, D, ws.glnf)
	layernormBackward(ws.glnf, x, w.LnFW, ws.lnfMean, ws.lnfRstd, T, D, ws.dx, g.LnFW, g.LnFB)

	grad := ws.dx
	for i := len(w.Layers) - 1; i >= 0; i-- {
		grad = layerBackward(&ws.blocks[i], &w.Layers[i], &g.Layers[i], ws, grad, scale)
	}
	for i := 0; i < T; i++ {
		id := int(text[i])
		gh := grad[i*D : (i+1)*D]
		ge := g.Emb[id*D : (id+1)*D]
		for d := 0; d < D; d++ {
			ge[d] += gh[d]
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
