// =====================================================================================
// ML LIBRARY (browser versions of the scikit-learn models used in the labs)
// =====================================================================================
function makeScaler(X) { const d = X[0].length, n = X.length, mu = new Array(d).fill(0), sd = new Array(d).fill(0);
  X.forEach(x => x.forEach((v, j) => mu[j] += v / n)); X.forEach(x => x.forEach((v, j) => sd[j] += (v - mu[j]) ** 2 / n));
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j]) || 1; return { mu, sd, t: x => x.map((v, j) => (v - mu[j]) / sd[j]) }; }
function shuffleIdx(n, rng) { const a = [...Array(n).keys()]; for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
function solveLin(A, b) { const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r; [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-12; for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / d; if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; } }
  return M.map((r, i) => r[n] / (r[i] || 1e-12)); }

class LinearModel { // ordinary least squares (degree 1) or polynomial degree 2, small ridge for stability
  constructor(o = {}) { this.degree = o.degree || 1; this.alpha = o.alpha ?? 1e-6; }
  feats(z) { if (this.degree === 1) return [1, ...z]; const f = [1, ...z]; for (let i = 0; i < z.length; i++) for (let j = i; j < z.length; j++) f.push(z[i] * z[j]); return f; }
  fit(X, y) { this.sc = makeScaler(X); const F = X.map(x => this.feats(this.sc.t(x))), p = F[0].length, A = Array.from({ length: p }, () => new Array(p).fill(0)), b = new Array(p).fill(0);
    F.forEach((f, i) => { for (let a = 0; a < p; a++) { b[a] += f[a] * y[i]; for (let c = a; c < p; c++) A[a][c] += f[a] * f[c]; } });
    for (let a = 0; a < p; a++) { for (let c = 0; c < a; c++) A[a][c] = A[c][a]; if (a) A[a][a] += this.alpha * X.length; }
    this.w = solveLin(A, b); return this; }
  predict(X) { return X.map(x => this.feats(this.sc.t(x)).reduce((s, v, i) => s + v * this.w[i], 0)); }
}

class KNNModel {
  constructor(o = {}) { this.k = o.k || 5; this.task = o.task || "reg"; }
  fit(X, y) { this.sc = makeScaler(X); this.Z = X.map(x => this.sc.t(x)); this.y = y; if (this.task === "clf") this.nc = Math.max(...y) + 1; return this; }
  _nb(x) { const z = this.sc.t(x), d = this.Z.map((r, i) => { let s = 0; for (let j = 0; j < z.length; j++) s += (r[j] - z[j]) ** 2; return [s, i]; });
    d.sort((a, b) => a[0] - b[0]); return d.slice(0, this.k).map(([s, i]) => [1 / (Math.sqrt(s) + 1e-9), i]); }
  predict(X) { if (this.task === "clf") return this.proba(X).map(p => p.indexOf(Math.max(...p)));
    return X.map(x => { const nb = this._nb(x); let a = 0, w = 0; nb.forEach(([wi, i]) => { a += wi * this.y[i]; w += wi; }); return a / w; }); }
  proba(X) { return X.map(x => { const p = new Array(this.nc).fill(0); let w = 0; this._nb(x).forEach(([wi, i]) => { p[this.y[i]] += wi; w += wi; }); return p.map(v => v / w); }); }
}

// ---- trees on binned inputs (histogram splits, like LightGBM / sklearn HistGradientBoosting) ----
function makeBins(X, nb = 48) { const d = X[0].length, edges = [];
  for (let j = 0; j < d; j++) { const u = [...new Set(X.map(x => x[j]))].sort((a, b) => a - b); let e;
    if (u.length <= nb) e = u.slice(0, -1).map((v, i) => (v + u[i + 1]) / 2);
    else { e = []; for (let q = 1; q < nb; q++) { const v = u[Math.floor(q * u.length / nb)], w = u[Math.floor(q * u.length / nb) - 1]; e.push((v + w) / 2); } e = [...new Set(e)]; }
    edges.push(e); }
  return edges; }
function binOf(v, e) { let lo = 0, hi = e.length; while (lo < hi) { const m = (lo + hi) >> 1; if (v <= e[m]) hi = m; else lo = m + 1; } return lo; }
function binX(X, edges) { return X.map(x => Uint8Array.from(x.map((v, j) => binOf(v, edges[j])))); }

class TreeModel {
  constructor(o = {}) { this.maxDepth = o.maxDepth ?? 8; this.minLeaf = o.minLeaf ?? 1; this.maxFeatures = o.maxFeatures ?? 1; this.task = o.task || "reg"; this.nc = o.nc || 0; this.rng = o.rng || seeded(o.seed || 1); this.random = !!o.random; }
  fit(X, y, opts = {}) { this.edges = opts.edges || makeBins(X); const B = opts.B || binX(X, this.edges), w = opts.w || null, idx = opts.idx || [...Array(X.length).keys()];
    if (this.task === "clf" && !this.nc) this.nc = Math.max(...y) + 1; this.d = X[0].length; this.imp = new Float64Array(this.d); this.root = this._grow(B, y, w, idx, 0); return this; }
  _leaf(y, w, idx) { if (this.task === "reg") { let s = 0, c = 0; idx.forEach(i => { const wi = w ? w[i] : 1; s += wi * y[i]; c += wi; }); return { v: s / c }; }
    const p = new Array(this.nc).fill(0); let c = 0; idx.forEach(i => { const wi = w ? w[i] : 1; p[y[i]] += wi; c += wi; }); return { p: p.map(v => v / c) }; }
  _grow(B, y, w, idx, depth) {
    if (depth >= this.maxDepth || idx.length < 2 * this.minLeaf) return this._leaf(y, w, idx);
    const d = this.d, nf = Math.max(1, Math.round(this.maxFeatures * d)), feats = nf < d ? shuffleIdx(d, this.rng).slice(0, nf) : [...Array(d).keys()];
    let best = null; const reg = this.task === "reg", K = this.nc;
    let tot = 0, totS = 0, totQ = 0; const totC = reg ? null : new Float64Array(K);
    idx.forEach(i => { const wi = w ? w[i] : 1; tot += wi; if (reg) { totS += wi * y[i]; totQ += wi * y[i] * y[i]; } else totC[y[i]] += wi; });
    const parentImp = reg ? totQ - totS * totS / tot : tot - [...totC].reduce((a, c) => a + c * c, 0) / tot;
    if (parentImp <= 1e-12) return this._leaf(y, w, idx);
    for (const j of feats) { const nb = this.edges[j].length + 1; if (nb < 2) continue;
      const cnt = new Float64Array(nb), n = new Int32Array(nb), S = reg ? new Float64Array(nb) : null, Q = reg ? new Float64Array(nb) : null, C = reg ? null : new Float64Array(nb * K);
      idx.forEach(i => { const b = B[i][j], wi = w ? w[i] : 1; cnt[b] += wi; n[b]++; if (reg) { S[b] += wi * y[i]; Q[b] += wi * y[i] * y[i]; } else C[b * K + y[i]] += wi; });
      let lw = 0, ln = 0, lS = 0, lQ = 0; const lC = reg ? null : new Float64Array(K);
      const cand = this.random ? [Math.floor(this.rng() * (nb - 1))] : null;
      for (let b = 0; b < nb - 1; b++) { lw += cnt[b]; ln += n[b]; if (reg) { lS += S[b]; lQ += Q[b]; } else for (let k = 0; k < K; k++) lC[k] += C[b * K + k];
        if (cand && b !== cand[0]) continue;
        const rw = tot - lw, rn = idx.length - ln; if (ln < this.minLeaf || rn < this.minLeaf || lw <= 0 || rw <= 0) continue;
        let imp; if (reg) { const rS = totS - lS, rQ = totQ - lQ; imp = (lQ - lS * lS / lw) + (rQ - rS * rS / rw); }
        else { let a = 0, c = 0; for (let k = 0; k < K; k++) { a += lC[k] * lC[k]; c += (totC[k] - lC[k]) ** 2; } imp = (lw - a / lw) + (rw - c / rw); }
        if (!best || imp < best.imp) best = { imp, j, b }; } }
    if (!best || best.imp >= parentImp - 1e-12) return this._leaf(y, w, idx);
    this.imp[best.j] += parentImp - best.imp;
    const L = [], R = []; idx.forEach(i => (B[i][best.j] <= best.b ? L : R).push(i));
    return { j: best.j, t: this.edges[best.j][best.b], l: this._grow(B, y, w, L, depth + 1), r: this._grow(B, y, w, R, depth + 1) }; }
  _walk(x) { let nd = this.root; while (nd.l) nd = x[nd.j] <= nd.t ? nd.l : nd.r; return nd; }
  predict(X) { return this.task === "reg" ? X.map(x => this._walk(x).v) : this.proba(X).map(p => p.indexOf(Math.max(...p))); }
  proba(X) { return X.map(x => this._walk(x).p); }
}

class ForestModel {
  constructor(o = {}) { this.n = o.nTrees || 100; this.o = o; this.task = o.task || "reg"; }
  fit(X, y, opt = {}) { const rng = seeded(this.o.seed || 7), edges = makeBins(X), B = binX(X, edges), n = X.length; this.nc = this.task === "clf" ? Math.max(...y) + 1 : 0; this.trees = [];
    for (let t = 0; t < this.n; t++) { const idx = Array.from({ length: n }, () => Math.floor(rng() * n));
      this.trees.push(new TreeModel({ ...this.o, nc: this.nc, rng, maxFeatures: this.o.maxFeatures ?? (this.task === "clf" ? Math.sqrt(X[0].length) / X[0].length : 1) }).fit(X, y, { edges, B, idx, w: opt.w })); }
    this.imp = new Float64Array(X[0].length); this.trees.forEach(t => t.imp.forEach((v, j) => this.imp[j] += v)); return this; }
  predict(X) { if (this.task === "clf") return this.proba(X).map(p => p.indexOf(Math.max(...p)));
    const s = new Float64Array(X.length); this.trees.forEach(t => t.predict(X).forEach((v, i) => s[i] += v / this.n)); return [...s]; }
  proba(X) { const P = X.map(() => new Array(this.nc).fill(0)); this.trees.forEach(t => t.proba(X).forEach((p, i) => p.forEach((v, k) => P[i][k] += v / this.n))); return P; }
}

class BoostModel { // gradient boosting for regression (squared error)
  constructor(o = {}) { this.n = o.nEst || 200; this.lr = o.lr || 0.1; this.depth = o.maxDepth || 3; this.sub = o.subsample || 1; this.seed = o.seed || 3; }
  fit(X, y) { const edges = makeBins(X), B = binX(X, edges), rng = seeded(this.seed); this.f0 = y.reduce((a, b) => a + b, 0) / y.length;
    const F = new Float64Array(y.length).fill(this.f0); this.trees = [];
    for (let m = 0; m < this.n; m++) { const r = y.map((v, i) => v - F[i]); let idx = [...Array(y.length).keys()]; if (this.sub < 1) idx = idx.filter(() => rng() < this.sub);
      const t = new TreeModel({ maxDepth: this.depth, minLeaf: 2, rng }).fit(X, r, { edges, B, idx }); this.trees.push(t); t.predict(X).forEach((v, i) => F[i] += this.lr * v); }
    this.imp = new Float64Array(X[0].length); this.trees.forEach(t => t.imp.forEach((v, j) => this.imp[j] += v)); return this; }
  predict(X) { const s = new Float64Array(X.length).fill(this.f0); this.trees.forEach(t => t.predict(X).forEach((v, i) => s[i] += this.lr * v)); return [...s]; }
}

class NetModel { // one hidden layer (tanh), Adam; regression (scaled target) or softmax classification
  constructor(o = {}) { this.h = o.hidden || 16; this.epochs = o.epochs || 300; this.lr = o.lr || 0.01; this.l2 = o.alpha ?? 1e-4; this.task = o.task || "reg"; this.seed = o.seed || 1; this.batch = o.batch || 32; }
  fit(X, y) { this.sc = makeScaler(X); const Z = X.map(x => this.sc.t(x)), rng = seeded(this.seed);
    if (this.task === "reg") { this.ym = y.reduce((a, b) => a + b, 0) / y.length; this.ys = Math.sqrt(y.reduce((a, b) => a + (b - this.ym) ** 2, 0) / y.length) || 1; this.Y = y.map(v => [(v - this.ym) / this.ys]); this.no = 1; }
    else { this.no = Math.max(...y) + 1; this.Y = y.map(c => { const t = new Array(this.no).fill(0); t[c] = 1; return t; }); }
    this.net = new MLP(Z[0].length, this.h, this.no, this.seed); this.net.softmax = this.task === "clf"; this.curve = [];
    for (let e = 0; e < this.epochs; e++) { this.net.epoch(Z, this.Y, this.lr, this.l2, this.batch, rng); if (e % Math.max(1, Math.floor(this.epochs / 40)) === 0 || e === this.epochs - 1) this.curve.push([e + 1, this.net.loss(Z, this.Y)]); }
    return this; }
  _out(X) { return X.map(x => this.net.predict(this.sc.t(x))); }
  predict(X) { if (this.task === "reg") return this._out(X).map(o => o[0] * this.ys + this.ym); return this.proba(X).map(p => p.indexOf(Math.max(...p))); }
  proba(X) { return this._out(X).map(o => softmax(o)); }
}
function softmax(o) { const m = Math.max(...o), e = o.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }

class MLP {
  constructor(nin, nh, nout, seed = 1) { this.nin = nin; this.nh = nh; this.nout = nout; const rng = seeded(seed);
    this.W1 = Float64Array.from({ length: nin * nh }, () => (rng() - .5) * 2 * Math.sqrt(3 / nin)); this.b1 = new Float64Array(nh);
    this.W2 = Float64Array.from({ length: nh * nout }, () => (rng() - .5) * 2 * Math.sqrt(3 / nh)); this.b2 = new Float64Array(nout);
    this.P = [this.W1, this.b1, this.W2, this.b2]; this.m = this.P.map(p => new Float64Array(p.length)); this.v = this.P.map(p => new Float64Array(p.length)); this.t = 0; }
  forward(x, h) { const { nin, nh, nout, W1, b1, W2, b2 } = this; for (let j = 0; j < nh; j++) { let a = b1[j]; for (let i = 0; i < nin; i++) a += x[i] * W1[i * nh + j]; h[j] = Math.tanh(a); }
    const y = new Float64Array(nout); for (let k = 0; k < nout; k++) { let a = b2[k]; for (let j = 0; j < nh; j++) a += h[j] * W2[j * nout + k]; y[k] = a; } return y; }
  predict(x) { return this.forward(x, new Float64Array(this.nh)); }
  loss(X, Y) { let s = 0; X.forEach((x, i) => { const o = this.predict(x);
    if (this.softmax) { const p = softmax([...o]); s -= Math.log(Math.max(1e-12, p[Y[i].indexOf(1)])); } else for (let k = 0; k < this.nout; k++) s += (o[k] - Y[i][k]) ** 2; });
    return s / X.length; }
  epoch(X, Y, lr, l2, batch, rng) { const n = X.length, idx = shuffleIdx(n, rng), { nin, nh, nout } = this, h = new Float64Array(nh), dh = new Float64Array(nh);
    for (let s0 = 0; s0 < n; s0 += batch) { const G = this.P.map(p => new Float64Array(p.length)), bs = Math.min(batch, n - s0);
      for (let q = s0; q < s0 + bs; q++) { const x = X[idx[q]], t = Y[idx[q]], o = this.forward(x, h), y = this.softmax ? softmax([...o]) : o; dh.fill(0);
        for (let k = 0; k < nout; k++) { const e = (y[k] - t[k]) / bs; G[3][k] += e; for (let j = 0; j < nh; j++) { G[2][j * nout + k] += e * h[j]; dh[j] += e * this.W2[j * nout + k]; } }
        for (let j = 0; j < nh; j++) { const d = dh[j] * (1 - h[j] * h[j]); G[1][j] += d; for (let i = 0; i < nin; i++) G[0][i * nh + j] += d * x[i]; } }
      this.t++; this.P.forEach((p, pi) => { const g = G[pi], m = this.m[pi], v = this.v[pi]; for (let i = 0; i < p.length; i++) { const gi = g[i] + (pi % 2 === 0 ? l2 * p[i] : 0);
        m[i] = .9 * m[i] + .1 * gi; v[i] = .999 * v[i] + .001 * gi * gi; p[i] -= lr * (m[i] / (1 - .9 ** this.t)) / (Math.sqrt(v[i] / (1 - .999 ** this.t)) + 1e-8); } }); } }
}

// ---- metrics ----
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
function regMetrics(y, p) { const m = mean(y); let ss = 0, st = 0, ae = 0, ape = 0, na = 0;
  y.forEach((v, i) => { const e = v - p[i]; ss += e * e; st += (v - m) ** 2; ae += Math.abs(e); if (Math.abs(v) > 1e-9) { ape += Math.abs(e / v); na++; } });
  return { R2: 1 - ss / st, MAE: ae / y.length, RMSE: Math.sqrt(ss / y.length), MAPE: na ? 100 * ape / na : NaN }; }
function clfMetrics(y, p, K) { const cm = Array.from({ length: K }, () => new Array(K).fill(0)); y.forEach((t, i) => cm[t][p[i]]++);
  const per = [...Array(K).keys()].map(k => { const tp = cm[k][k], fp = cm.reduce((s, r, i) => s + (i !== k ? r[k] : 0), 0), fn = cm[k].reduce((s, v, j) => s + (j !== k ? v : 0), 0), sup = cm[k].reduce((a, b) => a + b, 0);
    const pr = tp + fp ? tp / (tp + fp) : 0, rc = tp + fn ? tp / (tp + fn) : 0; return { precision: pr, recall: rc, f1: pr + rc ? 2 * pr * rc / (pr + rc) : 0, support: sup }; });
  const n = y.length, acc = y.filter((t, i) => t === p[i]).length / n, present = per.filter(c => c.support > 0);
  return { cm, per, accuracy: acc, f1_macro: mean(present.map(c => c.f1)), f1_micro: acc, f1_weighted: per.reduce((s, c) => s + c.f1 * c.support, 0) / n,
    precision_macro: mean(present.map(c => c.precision)), recall_macro: mean(present.map(c => c.recall)) }; }
function permImportance(model, X, y, score, rng, reps = 3) { const base = score(y, model.predict(X)), d = X[0].length, out = [];
  for (let j = 0; j < d; j++) { let drop = 0; for (let r = 0; r < reps; r++) { const col = shuffleIdx(X.length, rng).map(i => X[i][j]); const Xp = X.map((x, i) => { const z = [...x]; z[j] = col[i]; return z; }); drop += (base - score(y, model.predict(Xp))) / reps; } out.push(drop); }
  return out; }
