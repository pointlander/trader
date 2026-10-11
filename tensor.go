// Copyright 2026 The trader Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"simd"
)

func sqrt32(x float32) float32 { return float32(math.Sqrt(float64(x))) }

func dot(x, y []float32) float32 {
	n := len(x)
	if len(y) < n {
		panic("dot length")
	}
	var a simd.Float32s
	var i int
	for i = 0; i < n-a.Len()+1; i += a.Len() {
		u := simd.LoadFloat32s(x[i : i+a.Len()])
		v := simd.LoadFloat32s(y[i : i+a.Len()])
		a = u.MulAdd(v, a)
	}
	if i < n {
		u, _ := simd.LoadFloat32sPart(x[i:])
		v, _ := simd.LoadFloat32sPart(y[i:])
		a = u.MulAdd(v, a)
	}
	return sumSIMD(a)
}

func sumSIMD(x simd.Float32s) float32 {
	var buf [16]float32
	n := x.Len()
	if n > len(buf) {
		panic("simd width")
	}
	x.Store(buf[:n])
	var r float32
	for i := 0; i < n; i++ {
		r += buf[i]
	}
	return r
}

// axpy adds a*x to y. A zero scale leaves y unchanged.
func axpy(y, x []float32, a float32) {
	n := len(x)
	if len(y) < n {
		panic("axpy length")
	}
	if a == 0 || n == 0 {
		return
	}
	var lane simd.Float32s
	w := lane.Len()
	scale := simd.BroadcastFloat32s(a)
	i := 0
	for ; i+w <= n; i += w {
		yv := simd.LoadFloat32s(y[i : i+w])
		xv := simd.LoadFloat32s(x[i : i+w])
		xv.MulAdd(scale, yv).Store(y[i : i+w])
	}
	if i < n {
		yv, _ := simd.LoadFloat32sPart(y[i:n])
		xv, _ := simd.LoadFloat32sPart(x[i:n])
		xv.MulAdd(scale, yv).StorePart(y[i:n])
	}
}

// mul sets C = A @ B. A is [M, K], B is [K, N], C is [M, N], all row-major.
func mul(A []float32, M, K int, B []float32, N int, C []float32) {
	if len(A) < M*K || len(B) < K*N || len(C) < M*N {
		panic("mul length")
	}
	for i := 0; i < M; i++ {
		ai := A[i*K : i*K+K]
		ci := C[i*N : i*N+N]
		for j := 0; j < N; j++ {
			ci[j] = 0
		}
		for p := 0; p < K; p++ {
			a := ai[p]
			if a == 0 {
				continue
			}
			axpy(ci, B[p*N:p*N+N], a)
		}
	}
}

// mulAT sets C = A^T @ B. A is [K, M], B is [K, N], C is [M, N].
func mulAT(A []float32, K, M int, B []float32, N int, C []float32) {
	if len(A) < K*M || len(B) < K*N || len(C) < M*N {
		panic("mulAT length")
	}
	for i := 0; i < M*N; i++ {
		C[i] = 0
	}
	for k := 0; k < K; k++ {
		ak := A[k*M : k*M+M]
		bk := B[k*N : k*N+N]
		for m := 0; m < M; m++ {
			a := ak[m]
			if a == 0 {
				continue
			}
			axpy(C[m*N:m*N+N], bk, a)
		}
	}
}

// mulBT sets C = A @ B^T. A is [M, K], B is [N, K], C is [M, N].
func mulBT(A []float32, M, K int, B []float32, N int, C []float32) {
	if len(A) < M*K || len(B) < N*K || len(C) < M*N {
		panic("mulBT length")
	}
	for m := 0; m < M; m++ {
		am := A[m*K : m*K+K]
		cm := C[m*N : m*N+N]
		for n := 0; n < N; n++ {
			cm[n] = dot(am, B[n*K:n*K+K])
		}
	}
}

func addBias(X []float32, rows, cols int, b []float32) {
	for i := 0; i < rows; i++ {
		row := X[i*cols : i*cols+cols]
		for j := 0; j < cols; j++ {
			row[j] += b[j]
		}
	}
}

func sumRows(X []float32, rows, cols int, out []float32) {
	for i := 0; i < rows; i++ {
		row := X[i*cols : i*cols+cols]
		for j := 0; j < cols; j++ {
			out[j] += row[j]
		}
	}
}

const lnEps = 1e-5

func layernorm(x []float32, rows, cols int, w, b, y, mean, rstd []float32) {
	n := float32(cols)
	for i := 0; i < rows; i++ {
		row := x[i*cols : i*cols+cols]
		var sum float32
		for _, v := range row {
			sum += v
		}
		mu := sum / n
		var varr float32
		for _, v := range row {
			d := v - mu
			varr += d * d
		}
		varr /= n
		rs := 1 / sqrt32(varr+lnEps)
		mean[i] = mu
		rstd[i] = rs
		out := y[i*cols : i*cols+cols]
		for j, v := range row {
			out[j] = (v-mu)*rs*w[j] + b[j]
		}
	}
}

func layernormBackward(dy, x, w []float32, mean, rstd []float32, rows, cols int, dx, dw, db []float32) {
	n := float32(cols)
	invN := 1 / n
	for i := 0; i < rows; i++ {
		dyRow := dy[i*cols : i*cols+cols]
		xRow := x[i*cols : i*cols+cols]
		dxRow := dx[i*cols : i*cols+cols]
		mu := mean[i]
		rs := rstd[i]
		var sumDn, sumDnN float32
		for j := 0; j < cols; j++ {
			nrm := (xRow[j] - mu) * rs
			dn := dyRow[j] * w[j]
			sumDn += dn
			sumDnN += dn * nrm
			dw[j] += dyRow[j] * nrm
			db[j] += dyRow[j]
		}
		for j := 0; j < cols; j++ {
			nrm := (xRow[j] - mu) * rs
			dn := dyRow[j] * w[j]
			dxRow[j] = rs * (dn - sumDn*invN - nrm*sumDnN*invN)
		}
	}
}

func gelu(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Erf(float64(x)*0.7071067811865476)))
}

func dgelu(x float32) float32 {
	c := float32(math.Erf(float64(x) * 0.7071067811865476))
	phi := float32(math.Exp(float64(-0.5*x*x)) * 0.3989422804014327)
	return 0.5*(1+c) + x*phi
}

func applyGelu(x, y []float32) {
	for i, v := range x {
		y[i] = gelu(v)
	}
}

func applyDGelu(x, dy, dx []float32) {
	for i, v := range x {
		dx[i] = dgelu(v) * dy[i]
	}
}
