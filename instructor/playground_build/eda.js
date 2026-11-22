// =====================================================================================
// SHARED: exploratory data analysis (EDA), k-fold views, per-algorithm results, SHAP values
// =====================================================================================
const EDACFG = {
  asph: { inputs: () => ASPH_F, targets: ["asphaltene_wt_pct"], label: "asphaltene_wt_pct" },
  vfm: { inputs: () => VFM_F, targets: ["oil_rate_stb_d", "water_rate_bbl_d", "gas_rate_mmscf_d"] },
  eor: { inputs: () => EOR_F, targets: [], cls: "eor_method" } };
const skewOf = a => { const m = mean(a), s = sd(a) || 1; return mean(a.map(v => ((v - m) / s) ** 3)); };
function ranks(a) { const o = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]), r = new Array(a.length); let i = 0;
  while (i < o.length) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++; const rr = (i + j) / 2 + 1; for (let q = i; q <= j; q++) r[o[q][1]] = rr; i = j + 1; } return r; }
function pearson(a, b) { const ma = mean(a), mb = mean(b); let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; } return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0; }
const corrOf = (a, b, m) => m === "spearman" ? pearson(ranks(a), ranks(b)) : pearson(a, b);
const isNum = v => typeof v === "number" && isFinite(v);
// small multiples: n panels in a grid, returns layout pieces
function grid(n, cols, extra = {}) { const rows = Math.ceil(n / cols), L = { height: 70 + rows * 235, grid: { rows, columns: cols, pattern: "independent", xgap: .12, ygap: rows > 2 ? .32 : .26 }, showlegend: false, margin: { l: 46, r: 10, t: 26, b: 30 }, annotations: [] };
  for (let i = 0; i < n; i++) { const ax = i ? i + 1 : ""; L["xaxis" + ax] = { gridcolor: css("--plot-grid"), zerolinecolor: css("--plot-grid"), linecolor: css("--line"), tickfont: { size: 10 }, ...(extra.x ? extra.x(i) : {}) };
    L["yaxis" + ax] = { gridcolor: css("--plot-grid"), zerolinecolor: css("--plot-grid"), linecolor: css("--line"), tickfont: { size: 10 }, ...(extra.y ? extra.y(i) : {}) }; }
  return L; }
const axn = i => i ? String(i + 1) : "";
function titles(L, names) { names.forEach((t, i) => L.annotations.push({ text: `<b>${t}</b>`, xref: `x${axn(i)} domain`, yref: `y${axn(i)} domain`, x: .5, y: 1, yanchor: "bottom", yshift: 3, showarrow: false, font: { size: 11.5, color: css("--ink") } })); return L; }

// ---------------- EDA step ----------------
function edaControls(k) { const c = EDACFG[k];
  return (c.targets.length > 1 ? `<div class="f"><label for="${k}-eda-t">Target for the relation plots</label><select id="${k}-eda-t">${c.targets.map(t => `<option value="${t}">${t}</option>`).join("")}</select></div>` : "")
    + `<div class="f"><label for="${k}-eda-m">Correlation</label><select id="${k}-eda-m"><option value="pearson">Pearson (straight-line relation)</option><option value="spearman">Spearman (any rising or falling relation)</option></select></div>`
    + (k === "eor" ? `<label class="check"><input type="checkbox" id="${k}-eda-log" checked> Log scale for permeability and viscosity in the plots</label>` : ""); }
function edaRun(k) { const s = S[k], c = EDACFG[k], R = s.raw, inputs = c.inputs(), vars = [...inputs, ...c.targets], method = $(`${k}-eda-m`).value;
  const tsel = c.targets.length > 1 ? $(`${k}-eda-t`).value : c.targets[0];
  const stats = vars.map(v => { const all = R.map(r => r[v]), a = all.filter(isNum); return { variable: v, role: c.targets.includes(v) ? "target" : "input", count: a.length, missing: all.length - a.length, mean: mean(a), std: sd(a), min: Math.min(...a), Q1: quantile(a, .25), median: quantile(a, .5), Q3: quantile(a, .75), max: Math.max(...a), skew: skewOf(a), unique: new Set(a).size }; });
  const ok = R.filter(r => vars.every(v => isNum(r[v]))), cols = vars.map(v => ok.map(r => r[v]));
  const corr = vars.map((_, i) => vars.map((_, j) => i === j ? 1 : corrOf(cols[i], cols[j], method)));
  const dup = R.length - new Set(R.map(r => vars.map(v => r[v]).join("|") + (c.cls ? r[c.cls] : ""))).size;
  s.eda = { inputs, vars, targets: c.targets, stats, corr, method, tsel, dup, log: k === "eor" && $(`${k}-eda-log`).checked }; }
function edaRender(k) { const s = S[k], e = s.eda, c = EDACFG[k], R = s.raw, items = [], miss = e.stats.reduce((a, q) => a + q.missing, 0), V = e.vars, I = e.inputs;
  $(`${k}-met1`).innerHTML = met(R.length.toLocaleString(), "rows") + met(I.length, "inputs") + met(c.cls ? "1 (class)" : e.targets.length, c.cls ? "target" : e.targets.length > 1 ? "targets" : "target") + met(miss, "missing values", miss ? "warn" : "good") + met(e.dup, "identical rows", e.dup ? "warn" : "good");
  items.push(miss ? ["w", `${miss} missing values. Decide in step 3 whether to drop or fill them.`] : ["g", "No missing values in any column."]);
  const r = (i, j) => e.corr[i][j], name = e.method === "spearman" ? "Spearman" : "Pearson";
  if (!c.cls) e.targets.forEach(t => { const ti = V.indexOf(t), top = I.map((f, i) => [f, r(i, ti)]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 2);
    items.push(`<b>${t}</b> moves most with ${top.map(([f, v]) => `${f} (${name} r = ${v.toFixed(2)}, ${v > 0 ? "rises together" : "moves opposite"})`).join(" and ")}.`); });
  const pairs = []; for (let i = 0; i < I.length; i++) for (let j = i + 1; j < I.length; j++) if (Math.abs(r(i, j)) >= .85) pairs.push(`${I[i]} and ${I[j]} (r = ${r(i, j).toFixed(2)})`);
  if (pairs.length) items.push(["w", `Inputs that move together: ${pairs.join(", ")}. They carry overlapping information; linear models can get unstable coefficients, and importance gets shared between them.`]);
  if (!c.cls) { const weak = I.filter((f, i) => e.targets.every(t => Math.abs(r(i, V.indexOf(t))) < .05)); if (weak.length) items.push(["w", `${weak.join(", ")} ${weak.length > 1 ? "are" : "is"} almost unrelated to the target${e.targets.length > 1 ? "s" : ""} (|r| below 0.05). ${weak.includes("psep_psi") ? "Separator pressure is a test condition, not a well property: the rates do not depend on it." : "It may still matter in a non-linear way; the models will tell."}`]); }
  const sk = e.stats.filter(q => Math.abs(q.skew) > 1); if (sk.length) items.push(`Skewed: ${sk.map(q => `${q.variable} (skew ${q.skew.toFixed(1)})`).join(", ")}. A log transform often helps distance-based models and linear models.`);
  if (k === "vfm") items.push(["w", `${e.dup ? e.dup + " rows are exact copies, and m" : "M"}any more differ only in separator pressure: check this in step 3.`]);
  if (k === "eor") items.push("The box plots show which inputs separate the methods: thermal methods sit at high viscosity and low API, gas methods at light oil and depth.");
  $(`${k}-mean1`).innerHTML = meaning("What the data shows", items);
  // histograms
  const cols = V.length > 6 ? 4 : 3, Lh = titles(grid(V.length, cols, { x: i => ({ type: e.log && (V[i] === "permeability_md" || V[i] === "viscosity_cp") ? "log" : "linear" }) }), V);
  Plotly.react(`${k}-eda-hist`, V.map((v, i) => { const a = R.map(q => q[v]).filter(isNum), lg = e.log && (v === "permeability_md" || v === "viscosity_cp");
    return { x: lg ? a.map(x => Math.max(x, .1)) : a, type: "histogram", nbinsx: lg ? undefined : 30, xaxis: "x" + axn(i), yaxis: "y" + axn(i), marker: { color: e.targets.includes(v) ? TE() : TR(), line: { color: css("--surface"), width: .5 } } }; }),
    layout({ ...Lh, bargap: .04 }), CFG);
  $(`${k}-eda-hist-read`).textContent = `${V.length} variables, inputs in petrol and target${e.targets.length > 1 ? "s" : ""} in amber. Look for skew, gaps, spikes at one value and impossible values.`;
  // correlation heatmap
  const dark = document.documentElement.dataset.theme === "dark";
  Plotly.react(`${k}-eda-corr`, [{ z: e.corr, x: V, y: V, type: "heatmap", zmin: -1, zmax: 1, colorscale: dark ? [[0, "#E46A5E"], [.5, "#131E27"], [1, "#3DB9C5"]] : [[0, "#C0392B"], [.5, "#FFFFFF"], [1, "#0B6E79"]],
    text: e.corr.map(rw => rw.map(v => v.toFixed(2))), texttemplate: "%{text}", textfont: { size: 10.5 }, hovertemplate: "%{y} · %{x}: %{text}<extra></extra>", xgap: 1, ygap: 1, colorbar: { thickness: 10, len: .9 } }],
    layout({ margin: { l: 130, r: 10, t: 8, b: 110 }, xaxis: { tickangle: -40, tickfont: { size: 10.5 } }, yaxis: { autorange: "reversed", tickfont: { size: 10.5 } } }), CFG);
  $(`${k}-eda-corr-read`).textContent = `${name} r from −1 to +1. ${e.method === "spearman" ? "Spearman uses ranks, so it catches curved relations too." : "Pearson only sees straight-line relations; switch to Spearman for curved ones."}`;
  // standardised boxes
  Plotly.react(`${k}-eda-box`, V.map(v => { const a = R.map(q => q[v]).filter(isNum), lg = e.log && (v === "permeability_md" || v === "viscosity_cp"), b = lg ? a.map(x => Math.log10(Math.max(x, .1))) : a, m = mean(b), sdv = sd(b) || 1;
    return { y: b.map(x => (x - m) / sdv), type: "box", name: lg ? `log ${v}` : v, boxpoints: "outliers", marker: { size: 3, color: e.targets.includes(v) ? TE() : TR() }, line: { color: e.targets.includes(v) ? TE() : TR(), width: 1.3 } }; }),
    layout({ showlegend: false, margin: { l: 44, r: 8, t: 8, b: 110 }, xaxis: { tickangle: -40, tickfont: { size: 10.5 } }, yaxis: { title: { text: "standard deviations from the mean" } } }), CFG);
  // relation to target
  if (c.cls) { const Lb = titles(grid(I.length, 3, { y: i => ({ type: e.log && (I[i] === "permeability_md" || I[i] === "viscosity_cp") ? "log" : "linear" }), x: () => ({ tickfont: { size: 9 }, tickangle: -35 }) }), I), tr = [];
    I.forEach((f, i) => CLASSES.forEach((cl, j) => { const a = R.filter(q => q.eor_method === cl).map(q => q[f]).filter(isNum); tr.push({ y: e.log && (f === "permeability_md" || f === "viscosity_cp") ? a.map(x => Math.max(x, .1)) : a, x: a.map(() => SHORT[j]), type: "box", name: cl, xaxis: "x" + axn(i), yaxis: "y" + axn(i), marker: { color: CCOL[j], size: 2.5 }, line: { width: 1.1 }, boxpoints: "outliers" }); }));
    Plotly.react(`${k}-eda-tgt`, tr, layout({ ...Lb, margin: { l: 46, r: 10, t: 26, b: 46 } }), CFG); $(`${k}-eda-tgt-read`).textContent = "Each panel: one input, one box per EOR method. Inputs whose boxes do not overlap are the ones that separate the methods."; }
  else { const t = e.tsel, Ls = titles(grid(I.length, I.length > 4 ? 3 : I.length, {}), I.map(f => `${f}  (r = ${r(V.indexOf(f), V.indexOf(t)).toFixed(2)})`));
    Plotly.react(`${k}-eda-tgt`, I.map((f, i) => ({ x: R.map(q => q[f]), y: R.map(q => q[t]), mode: "markers", type: "scatter", xaxis: "x" + axn(i), yaxis: "y" + axn(i), marker: { size: 3.5, color: TR(), opacity: .45 }, hovertemplate: `${f} %{x}<br>${t} %{y}<extra></extra>` })), layout(Ls), CFG);
    $(`${k}-eda-tgt-read`).textContent = `Every input against ${t}. A clear band means a strong relation; a cloud means the input says little on its own.`; }
  $(`${k}-eda-tab`).innerHTML = tableHTML(e.stats.map(q => ({ variable: q.variable, role: q.role, count: q.count, missing: q.missing, mean: fmt(q.mean), std: fmt(q.std), min: fmt(q.min), Q1: fmt(q.Q1), median: fmt(q.median), Q3: fmt(q.Q3), max: fmt(q.max), skew: +q.skew.toFixed(2), unique: q.unique })), 30); }
function edaDC(k) { const e = S[k].eda; return { tables: { "summary_statistics.csv": e.stats.map(q => Object.fromEntries(Object.entries(q).map(([a, b]) => [a, r4(b)]))), "correlation_matrix.csv": e.vars.map((v, i) => ({ variable: v, ...Object.fromEntries(e.vars.map((w, j) => [w, r4(e.corr[i][j])])) })) }, dataNote: `${e.method} correlation, on the data as loaded (before cleaning)` }; }
function pyEda(k) { const e = S[k].eda, c = EDACFG[k], V = e.vars, I = e.inputs;
  let out = `\n# ---- Step 2 · explore the data (EDA) ----
cols = ${py(V)}
stats = df[cols].describe().T
stats["missing"] = df[cols].isna().sum(); stats["skew"] = df[cols].skew(); stats["unique"] = df[cols].nunique()
print(stats.round(3))
print("identical rows:", df.duplicated(subset=cols${c.cls ? ` + [${py(c.cls)}]` : ""}).sum())
corr = df[cols].corr(method=${py(e.method)})
fig, ax = plt.subplots(figsize=(8, 6.5)); im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right"); ax.set_yticks(range(len(cols)), cols)
for i in range(len(cols)):
    for j in range(len(cols)): ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, label="${e.method} r"); plt.title("Correlation matrix"); plt.tight_layout(); plt.show()
df[cols].hist(bins=30, figsize=(13, 8)); plt.tight_layout(); plt.show()
z = (df[cols] - df[cols].mean()) / df[cols].std()
z.boxplot(figsize=(12, 4), rot=40); plt.ylabel("standard deviations from the mean"); plt.tight_layout(); plt.show()`;
  if (c.cls) out += `\nfig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), ${py(I)}):
    df.boxplot(column=col, by=${py(c.cls)}, ax=ax, rot=35)${e.log ? `\n    if col in ("permeability_md", "viscosity_cp"): ax.set_yscale("log")` : ""}
plt.suptitle("Each input by EOR method"); plt.tight_layout(); plt.show()`;
  else out += `\ninputs, target = ${py(I)}, ${py(e.tsel)}
fig, axes = plt.subplots(${Math.ceil(I.length / 3)}, 3, figsize=(14, ${Math.ceil(I.length / 3) * 3.6}))
for ax, col in zip(axes.ravel(), inputs):
    ax.scatter(df[col], df[target], s=6, alpha=0.4); ax.set_xlabel(col); ax.set_ylabel(target)
    ax.set_title(f"r = {df[col].corr(df[target], method=${py(e.method)}):.2f}")
for ax in axes.ravel()[len(inputs):]: ax.axis("off")
plt.tight_layout(); plt.show()`;
  return out; }
function edaStage(k) { const clf = k === "eor";
  return { title: "Explore the data (EDA)", what: "Look before you clean or model: how each variable is distributed, how variables relate to each other and to the target, and the summary statistics.",
    controls: edaControls(k), runLabel: "Explore the data", empty: "Press <b>Explore the data</b>. Nothing is changed in this step; it only looks.", run: edaRun, render: edaRender,
    dc: () => ({ ...edaDC(k), code: k === "eor" ? pyEor("eda") : pyReg(k, "eda"), codeNote: pyNote(1), codeFile: `${k}_workflow_step2.py` }),
    charts: [{ id: `${k}-eda-hist`, title: "Distribution of every variable", read: "", cls: "auto", span: true },
      { id: `${k}-eda-corr`, title: "Correlation matrix", read: "", cls: "tall" },
      { id: `${k}-eda-box`, title: "Spread and outliers", read: "Every variable scaled to mean 0 and standard deviation 1. Dots beyond the whiskers are candidates for the cleaning step.", cls: "tall" },
      { id: `${k}-eda-tgt`, title: clf ? "Each input by EOR method" : "Each input against the target", read: "", cls: "auto", span: true },
      { id: `${k}-eda-tab-c`, title: "Summary statistics", read: "count, mean, standard deviation, quartiles, skewness and the number of different values", html: `<div class="scroll" id="${k}-eda-tab"></div>`, span: true }],
    task: clf ? ["Which two inputs separate thermal from gas methods best?", "Which inputs are so skewed that a log scale is needed?"]
      : k === "vfm" ? ["Which gauge moves most with the oil rate? Does that make physical sense?", "Separator pressure barely correlates with any rate. Why?"] : ["Which input relates most to precipitation: Pearson or Spearman, and why do they differ?", "Do you see values that look out of place before any cleaning?"] }; }

// ---------------- k-fold views and per-algorithm results ----------------
function foldDiagram(id, fold, K, groupWord) { const n = fold.length, ord = [...Array(n).keys()].sort((a, b) => fold[a] - fold[b] || a - b), step = Math.max(1, Math.ceil(n / 600)), cols = ord.filter((_, j) => j % step === 0);
  const z = Array.from({ length: K }, (_, f) => cols.map(i => fold[i] === f ? 1 : 0));
  Plotly.react(id, [{ z, x: cols.map((_, j) => j * step + 1), y: Array.from({ length: K }, (_, f) => `fold ${f + 1}`), type: "heatmap", showscale: false, colorscale: [[0, TR()], [1, TE()]], xgap: 0, ygap: 3, hovertemplate: "%{y}, training row %{x}: %{z}<extra>1 = scored, 0 = trains</extra>" }],
    layout({ margin: { l: 64, r: 10, t: 8, b: 40 }, xaxis: { title: { text: `training rows, sorted by the fold that scores them${groupWord ? ` (whole ${groupWord} stay together)` : ""}` }, showgrid: false }, yaxis: { autorange: "reversed", showgrid: false } }), CFG); }
function foldLines(id, res, sel, names, metric) { const K = Math.max(...res.map(r => r.folds.length)), xs = Array.from({ length: K }, (_, i) => `fold ${i + 1}`);
  Plotly.react(id, res.filter(r => r.folds.length).map((r, i) => ({ x: xs, y: r.folds, name: names[r.alg], mode: "lines+markers", type: "scatter", line: { width: r.alg === sel ? 3.2 : 1.3, color: r.alg === sel ? TE() : CCOL[i % 10] }, marker: { size: r.alg === sel ? 9 : 5 }, opacity: r.alg === sel ? 1 : .55 })),
    layout({ legend: { orientation: "h", y: 1.2, x: 0, font: { size: 10.5 } }, yaxis: { title: { text: metric } } }), CFG); }
function algButtons(k, res, sel, names) { $(`${k}-algbtns`).innerHTML = res.map(r => `<button type="button" class="btn sm ${r.alg === sel ? "" : "alt"}" data-alg="${r.alg}" aria-pressed="${r.alg === sel}">${names[r.alg]}</button>`).join(""); }
function algWire(k, draw, finalSel) { $(`${k}-algbtns`).addEventListener("click", e => { const b = e.target.closest("[data-alg]"); if (!b || !S[k] || !S[k].model) return; S[k].model.sel = b.dataset.alg; draw(k); });
  $(`${k}-algfinal`).addEventListener("click", () => { const m = S[k] && S[k].model; if (!m || !m.sel || m.sel === "taber") return; $(finalSel).value = m.sel; $(finalSel).dispatchEvent(new Event("change", { bubbles: true })); $(`${k}-run4`).click(); }); }
function cvClick(k, id, names, draw) { const el = $(id); if (el._wired) return; el._wired = true;
  el.on("plotly_click", ev => { const lab = ev.points && ev.points[0] && ev.points[0].y, alg = Object.keys(names).find(a => names[a] === lab) || (lab && lab.startsWith("Taber") ? "taber" : null); if (alg && S[k].model.cv.some(r => r.alg === alg)) { S[k].model.sel = alg; draw(k); } }); }

// ---------------- SHAP (permutation sampling, Strumbelj & Kononenko; antithetic orderings) ----------------
async function shapValues(f, X, B, M, seed, progress) { const n = X.length, p = X[0].length, rng = seeded(seed), rows = [], plan = [];
  for (let i = 0; i < n; i++) for (let m = 0; m < M; m++) { const z = B[Math.floor(rng() * B.length)], ord = shuffleIdx(p, rng);
    [ord, [...ord].reverse()].forEach(o => { const cur = z.slice(); plan.push([i, o, rows.length]); rows.push(cur.slice()); o.forEach(j => { cur[j] = X[i][j]; rows.push(cur.slice()); }); }); }
  const out = [], CH = 4000; for (let a = 0; a < rows.length; a += CH) { out.push(...f(rows.slice(a, a + CH))); if (progress) await progress(Math.min(1, (a + CH) / rows.length)); }
  const O = out[0].length, phi = Array.from({ length: n }, () => Array.from({ length: p }, () => new Array(O).fill(0)));
  plan.forEach(([i, o, s]) => o.forEach((j, t) => { for (let q = 0; q < O; q++) phi[i][j][q] += (out[s + t + 1][q] - out[s + t][q]) / (2 * M); }));
  const fb = f(B), base = Array.from({ length: O }, (_, q) => mean(fb.map(v => v[q]))), fx = f(X);
  return { phi, base, fx }; }
const shapPick = (sh, q) => sh.phi.map(r => r.map(v => v[q]));
function drawBeeswarm(id, phi, X, names) { const p = names.length, imp = names.map((_, j) => mean(phi.map(r => Math.abs(r[j])))), ord = [...Array(p).keys()].sort((a, b) => imp[a] - imp[b]), rng = seeded(3), xs = [], ys = [], cs = [], tx = [];
  ord.forEach((j, row) => { const v = X.map(x => x[j]), lo = Math.min(...v), hi = Math.max(...v);
    phi.forEach((r, i) => { xs.push(r[j]); ys.push(row + (rng() - .5) * .62); cs.push(hi > lo ? (X[i][j] - lo) / (hi - lo) : .5); tx.push(`${names[j]} = ${fmt(X[i][j])}<br>SHAP ${fmt(r[j])}`); }); });
  Plotly.react(id, [{ x: xs, y: ys, mode: "markers", type: "scattergl", text: tx, hovertemplate: "%{text}<extra></extra>", marker: { size: 5.5, color: cs, colorscale: [[0, "#2F7FD8"], [.5, "#B9A6D6"], [1, "#D9534F"]], cmin: 0, cmax: 1, opacity: .85, colorbar: { title: { text: "input value", side: "right" }, tickvals: [0, 1], ticktext: ["low", "high"], thickness: 10, len: .8 } } }],
    layout({ margin: { l: 150, r: 10, t: 8, b: 44 }, xaxis: { title: { text: "SHAP value: push on the prediction (target units)" }, zeroline: true, zerolinecolor: css("--muted") }, yaxis: { tickvals: ord.map((_, r) => r), ticktext: ord.map(j => names[j]), range: [-.6, p - .4] } }), CFG);
  return imp; }
function drawShapBar(id, imp, names, unit) { const ord = [...imp.keys()].sort((a, b) => imp[a] - imp[b]);
  Plotly.react(id, [{ y: ord.map(j => names[j]), x: ord.map(j => imp[j]), type: "bar", orientation: "h", marker: { color: TE() }, text: ord.map(j => fmt(imp[j])), textposition: "auto" }], layout({ margin: { l: 150, r: 14, t: 8, b: 44 }, xaxis: { title: { text: `mean |SHAP| (${unit})` } } }), CFG); }
function drawShapDep(id, phi, X, names, imp) { const j = imp.indexOf(Math.max(...imp)), j2 = imp.map((v, q) => q === j ? -1 : v).indexOf(Math.max(...imp.map((v, q) => q === j ? -1 : v)));
  Plotly.react(id, [{ x: X.map(x => x[j]), y: phi.map(r => r[j]), mode: "markers", type: "scatter", marker: { size: 6, color: j2 >= 0 ? X.map(x => x[j2]) : TR(), colorscale: "Viridis", colorbar: j2 >= 0 ? { title: { text: names[j2], side: "right" }, thickness: 10, len: .8 } : undefined, opacity: .85 }, hovertemplate: `${names[j]} %{x}<br>SHAP %{y:.4g}<extra></extra>` }],
    layout({ xaxis: { title: { text: names[j] } }, yaxis: { title: { text: `SHAP value of ${names[j]}` } } }), CFG); return [names[j], j2 >= 0 ? names[j2] : ""]; }
function shapOne(f, x, B, M, seed) { const p = x.length, rng = seeded(seed), rows = [], plan = [];
  for (let m = 0; m < M; m++) { const z = B[Math.floor(rng() * B.length)], ord = shuffleIdx(p, rng); [ord, [...ord].reverse()].forEach(o => { const cur = z.slice(); plan.push([o, rows.length]); rows.push(cur.slice()); o.forEach(j => { cur[j] = x[j]; rows.push(cur.slice()); }); }); }
  const out = f(rows), phi = new Array(p).fill(0); plan.forEach(([o, s]) => o.forEach((j, t) => phi[j] += (out[s + t + 1] - out[s + t]) / (2 * M)));
  return { phi, base: mean(f(B)), fx: f([x])[0] }; }
function drawWaterfall(id, sh, names, x, unit) { const ord = [...names.keys()].sort((a, b) => Math.abs(sh.phi[b]) - Math.abs(sh.phi[a]));
  Plotly.react(id, [{ type: "waterfall", orientation: "h", measure: ["absolute", ...ord.map(() => "relative"), "total"], y: ["average prediction", ...ord.map(j => `${names[j]} = ${fmt(x[j])}`), "this prediction"], x: [sh.base, ...ord.map(j => sh.phi[j]), 0],
    text: [fmt(sh.base), ...ord.map(j => (sh.phi[j] >= 0 ? "+" : "") + fmt(sh.phi[j])), fmt(sh.fx)], textposition: "outside", connector: { line: { color: css("--line") } },
    increasing: { marker: { color: "#D9534F" } }, decreasing: { marker: { color: "#2F7FD8" } }, totals: { marker: { color: TE() } } }],
    layout({ margin: { l: 190, r: 40, t: 8, b: 44 }, xaxis: { title: { text: unit } }, yaxis: { autorange: "reversed" }, showlegend: false }), CFG); }
const pyShapNote = `# SHAP values (pip install shap). The browser uses the same permutation method with fewer samples, so values are close, not identical.`;
