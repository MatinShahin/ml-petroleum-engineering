// =====================================================================================
// DATA
// =====================================================================================
const D = { asph: parseCSV(window.CSV_ASPH), eor: parseCSV(window.CSV_EOR), vfm: parseCSV(window.CSV_VFM) };
D.asph.forEach(r => { r.series = `${r.temperature_K}K C${r.alkane_carbon_number}`; r.suspect = r.temperature_K === 332 && r.alkane_carbon_number <= 8 ? "suspect" : "normal"; });
D.vfm.forEach(r => { r.case_id = [r.choke_in, r.bhp_psi, r.whp_psi, r.wht_F, r.tsep_F].join("|"); });
const CASES = [...new Set(D.vfm.map(r => r.case_id))]; D.vfm.forEach(r => r.case_id = CASES.indexOf(r.case_id));
const ASPH_F = ["temperature_K", "alkane_carbon_number", "dilution_ratio"], EOR_F = ["permeability_md", "depth_ft", "api_gravity", "viscosity_cp", "temperature_F", "oil_saturation_pct"];
const VFM_F = ["choke_in", "bhp_psi", "whp_psi", "wht_F", "psep_psi", "tsep_F"], CLASSES = ["CO2", "Combustion", "Hydrocarbon", "Nitrogen", "Polymer", "Steam Injection"];
const CCOL = ["#2F7FD8", "#D9534F", "#D9940F", "#8E5BD6", "#1E9E6A", "#14A3B8", "#C2508F", "#6E9A1E", "#7A8794", "#E0702B"];
const fmt = (v, d = 3) => typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toFixed(0) : +v.toFixed(d)) : v;
const numCols = rows => Object.keys(rows[0]).filter(c => typeof rows[0][c] === "number");
function quantile(a, q) { const s = [...a].sort((x, y) => x - y), p = (s.length - 1) * q, lo = Math.floor(p); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (p - lo); }
function iqr(vals, k) { const q1 = quantile(vals, .25), q3 = quantile(vals, .75), d = q3 - q1; return [q1 - k * d, q3 + k * d]; }
async function progress(id, f) { $(id).style.width = (100 * f).toFixed(0) + "%"; await sleep(); }

// ---- Taber screening rules (same as aiml/generators.py) ----
const TABER = { "Nitrogen": { api: [35, 99], visc: [0, .4], depth: [6000, 99999], sat: [40, 100] }, "Hydrocarbon": { api: [23, 99], visc: [0, 3], depth: [4000, 99999], sat: [30, 100] },
  "CO2": { api: [22, 99], visc: [0, 10], depth: [2500, 99999], sat: [20, 100] }, "Polymer": { api: [15, 99], visc: [10, 150], depth: [0, 9000], perm: [10, 1e9], temp: [0, 200], sat: [50, 100] },
  "Combustion": { api: [10, 27], visc: [0, 5000], depth: [0, 11500], perm: [50, 1e9], sat: [50, 100] }, "Steam Injection": { api: [8, 25], visc: [20, 2e5], depth: [0, 4500], perm: [200, 1e9], sat: [40, 100] } };
const TKEY = { api: "api_gravity", visc: "viscosity_cp", depth: "depth_ft", perm: "permeability_md", temp: "temperature_F", sat: "oil_saturation_pct" };
const TORDER = ["Polymer", "Steam Injection", "Nitrogen", "CO2", "Hydrocarbon", "Combustion"];
function taberScores(r) { const o = {}; for (const m of TORDER) { const c = Object.entries(TABER[m]); o[m] = c.filter(([k, [lo, hi]]) => r[TKEY[k]] >= lo && r[TKEY[k]] <= hi).length / c.length; } return o; }
function taberScreen(r) { const s = taberScores(r), full = TORDER.find(m => s[m] >= .999); if (full) return full; let b = TORDER[0]; TORDER.forEach(m => { if (s[m] > s[b]) b = m; }); return b; }

// ---- generators (same physics as aiml/generators.py; the browser uses its own random numbers) ----
const asphModel = (T, C, DR) => Math.max(.1, 7.31 - .371 * (C - 5) - .0651 * (T - 293)) * (1 - Math.exp(-Math.max(DR - 1.36, 0) / 7.43));
function genAsph(o) { const r = seeded(o.seed + 11), rows = [], drs = Array.from({ length: o.ndr }, (_, i) => 3.6 + (24.4 - 3.6) * i / (o.ndr - 1));
  for (const T of [293, 308, 323, 332]) for (const C of [5, 6, 7, 8, 9, 10, 12]) for (const dr0 of drs) { const DR = dr0 * (1 + .01 * gauss(r)); let w = asphModel(T, C, DR) * (1 + o.noise * gauss(r)); const bad = r() < o.out;
    if (bad) w *= o.fac * (0.8 + 0.4 * r()); rows.push({ temperature_K: T, alkane_carbon_number: C, dilution_ratio: +DR.toFixed(4), asphaltene_wt_pct: +Math.max(0, w).toFixed(4), is_outlier: bad ? 1 : 0 }); }
  return rows; }
function vfmWell(c, pr, pi, wc, glr, depth, psep) { const S = c * 64, grad = .433 * (.85 + .15 * wc) * (.45 + .35 / (1 + glr));
  const pwhOf = q => Math.max(435 * glr ** .546 * q / S ** 1.89, psep + 150 + 2e-5 * q * q), res = q => pi * (pr - (pwhOf(q) + grad * depth + 2.5e-6 * q * q * depth / 1e3)) - q;
  let lo = 1, hi = pi * pr; for (let it = 0; it < 80; it++) { const m = (lo + hi) / 2; if (res(m) > 0) lo = m; else hi = m; } const q = (lo + hi) / 2;
  return { q, pwf: pr - q / pi, pwh: pwhOf(q), wht: 60 + 130 * (1 - Math.exp(-q / 2500)) }; }
function genVfm(o) { const r = seeded(o.seed + 5), chokes = [.25, .32, .42, .54, .7, .9, 1.16, 1.5, 1.94, 2.5], rows = [];
  for (let i = 0; i < o.n; i++) { const c = chokes[Math.floor(r() * 10)], pr = 5300 + 400 * r(), pi = 3.5 + 3 * r(), wc = o.disc ? Math.min(o.wcmax, .2 * Math.floor(r() * 5)) : o.wcmax * r(),
      gor = Math.max(200, o.gor * (0.6 + 0.8 * r())), ps = 100 * (1 + Math.floor(r() * 6)), glr = Math.max(.05, gor * (1 - wc) / 1000), w = vfmWell(c, pr, pi, wc, glr, o.depth, ps);
    const e = v => v * (1 + o.noise * gauss(r)), tsep = 60 + .45 * (w.wht - 60) * (1 - Math.exp(-w.q / 3000));
    rows.push({ test_id: i, choke_in: c, bhp_psi: +e(w.pwf).toFixed(2), whp_psi: +e(w.pwh).toFixed(2), wht_F: +e(w.wht).toFixed(2), psep_psi: ps, tsep_F: +e(tsep).toFixed(2),
      oil_rate_stb_d: +(w.q * (1 - wc)).toFixed(1), water_rate_bbl_d: +(w.q * wc).toFixed(1), gas_rate_mmscf_d: +(w.q * (1 - wc) * gor / 1e6).toFixed(3), water_cut: +wc.toFixed(4) }); }
  return rows; }
function genEor(o) { const r = seeded(o.seed + 3), rows = [], by = {}; D.eor.forEach(p => (by[p.eor_method] = by[p.eor_method] || []).push(p));
  for (const [m, g] of Object.entries(by)) { const k = Math.max(3, Math.round(o.n * g.length / D.eor.length));
    const st = EOR_F.map(f => { const l = g.map(p => Math.log(Math.max(p[f], .1))), mu = mean(l), sd = Math.sqrt(mean(l.map(v => (v - mu) ** 2))) || .3; return [mu, sd + o.noise]; });
    for (let i = 0; i < k; i++) { const p = {}; EOR_F.forEach((f, j) => p[f] = Math.exp(st[j][0] + st[j][1] * gauss(r))); p.oil_saturation_pct = Math.min(98, Math.max(5, p.oil_saturation_pct));
      ["permeability_md", "api_gravity", "temperature_F", "oil_saturation_pct"].forEach(f => p[f] = +p[f].toFixed(1)); p.depth_ft = Math.round(p.depth_ft); p.viscosity_cp = +p.viscosity_cp.toFixed(2);
      p.eor_method = o.label === "taber" ? taberScreen(p) : m; rows.push(p); } }
  return shuffleIdx(rows.length, r).map(i => rows[i]); }

