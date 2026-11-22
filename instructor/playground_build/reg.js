// =====================================================================================
// REGRESSION PROJECTS: asphaltene precipitation (1 target) and virtual flow metering (3 targets)
// =====================================================================================
const S = {};
const ALGN = { lin: "Linear regression", poly: "Polynomial (degree 2)", knn: "K-nearest neighbours", tree: "Decision tree", rf: "Random forest", gb: "Gradient boosting", nn: "Neural network (MLP)" };
const AIML = { lin: "Linear regression", poly: "Polynomial (deg 2)", knn: "KNN", tree: "Decision tree", rf: "Random forest", gb: "Gradient boosting", nn: "Neural network (MLP)" };
const HP = {
  lin: [], poly: [], knn: [["k", "Neighbours k", 1, 30, 1, 3]],
  tree: [["depth", "Max depth", 1, 16, 1, 8], ["leaf", "Min samples per leaf", 1, 20, 1, 1]],
  rf: [["trees", "Trees", 10, 300, 10, 100], ["depth", "Max depth", 2, 20, 1, 12], ["leaf", "Min samples per leaf", 1, 10, 1, 1]],
  gb: [["trees", "Boosting rounds", 10, 600, 10, 200], ["lr", "Learning rate", .01, .5, .01, .1], ["depth", "Tree depth", 1, 8, 1, 3]],
  nn: [["hidden", "Hidden neurons", 2, 64, 1, 16], ["epochs", "Epochs", 20, 1500, 20, 400], ["lr", "Learning rate", .001, .05, .001, .01], ["alpha", "L2 penalty (alpha)", 0, .05, .001, .0001]] };
const HPDEF = (alg, big) => Object.fromEntries(HP[alg].map(([id, , , , , d]) => [id, d])), BIGFIX = { rf: { trees: 30, depth: 14 }, nn: { epochs: 80, hidden: 24 }, gb: { trees: 150, depth: 4 }, knn: { k: 5 } };
function hpHTML(pre, alg, big) { const d = { ...HPDEF(alg), ...(big ? BIGFIX[alg] || {} : {}) };
  return HP[alg].length ? HP[alg].map(([id, l, mn, mx, st]) => rangeCtl(`${pre}-${id}`, l, mn, mx, st, d[id])).join("") : `<p class="hint">No settings: least squares has one exact answer.</p>`; }
function hpBind(pre, alg) { HP[alg].forEach(([id, , , , st]) => bind(`${pre}-${id}`, v => st < 1 ? String(+v) : v)); }
const hpRead = (pre, alg) => Object.fromEntries(HP[alg].map(([id]) => [id, +$(`${pre}-${id}`).value]));
function makeModel(alg, h, task, seed, nc) {
  if (alg === "lin") return new LinearModel(); if (alg === "poly") return new LinearModel({ degree: 2, alpha: 1e-6 });
  if (alg === "knn") return new KNNModel({ k: h.k, task }); if (alg === "tree") return new TreeModel({ maxDepth: h.depth, minLeaf: h.leaf, task, nc, seed: seed + 1 });
  if (alg === "rf") return new ForestModel({ nTrees: h.trees, maxDepth: h.depth, minLeaf: h.leaf, task, seed: seed + 7 });
  if (alg === "gb") return new BoostModel({ nEst: h.trees, lr: h.lr, maxDepth: h.depth, seed: seed + 3 });
  return new NetModel({ hidden: h.hidden, epochs: h.epochs, lr: h.lr, alpha: h.alpha, task, seed: seed + 1 }); }
const PYREG = { lin: h => "make_pipeline(StandardScaler(), LinearRegression())", poly: h => "make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1e-6))",
  knn: h => `make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=${h.k}, weights="distance"))`, tree: (h, s) => `DecisionTreeRegressor(max_depth=${h.depth}, min_samples_leaf=${h.leaf}, random_state=${s})`,
  rf: (h, s) => `RandomForestRegressor(n_estimators=${h.trees}, max_depth=${h.depth}, min_samples_leaf=${h.leaf}, random_state=${s})`,
  gb: (h, s) => `GradientBoostingRegressor(n_estimators=${h.trees}, learning_rate=${py(h.lr)}, max_depth=${h.depth}, random_state=${s})`,
  nn: (h, s) => `TransformedTargetRegressor(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(${h.hidden},), activation="tanh",
        alpha=${py(h.alpha)}, learning_rate_init=${py(h.lr)}, max_iter=${h.epochs}, batch_size=32, random_state=${s})), transformer=StandardScaler())` };
const KIT1 = `import sys, os
sys.path.insert(0, "aiml_course_kit")            # the folder where you unzipped the course kit
import numpy as np, pandas as pd, matplotlib.pyplot as plt`;
const PYIMP = `from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone`;
const r4 = v => typeof v === "number" ? +v.toFixed(4) : v;

// ---------------- project definitions ----------------
const REGCFG = {
  asph: { targets: ["asphaltene_wt_pct"], units: { asphaltene_wt_pct: "wt%" }, short: { asphaltene_wt_pct: "precipitate" }, base: ASPH_F,
    extra: { log_dilution: r => Math.log(r.dilution_ratio), inv_temperature: r => 1000 / r.temperature_K },
    group: r => `${r.temperature_K}K C${r.alkane_carbon_number}`, groupName: "series", groupPlural: "series", groupWord: "unseen series (a temperature + n-alkane curve)", hold: ["temperature_K", "alkane_carbon_number"], big: false },
  vfm: { targets: ["oil_rate_stb_d", "water_rate_bbl_d", "gas_rate_mmscf_d"], units: { oil_rate_stb_d: "STB/d", water_rate_bbl_d: "bbl/d", gas_rate_mmscf_d: "MMscf/d" }, short: { oil_rate_stb_d: "oil", water_rate_bbl_d: "water", gas_rate_mmscf_d: "gas" },
    base: VFM_F, extra: {}, group: r => r.case_id, groupName: "well test", groupPlural: "well tests", groupWord: "unseen well tests (all copies of a test stay together)", hold: ["choke_in"], big: true } };
const fx = (k, F) => r => F.map(f => REGCFG[k].extra[f] ? REGCFG[k].extra[f](r) : r[f]);
const median = a => quantile(a, .5);

// ---------- step 1 · data ----------
function regDataControls(k) { return (k === "asph" ? srcToggle(k) : "") + (k === "asph"
  ? `<div id="asph-coursebox"><p class="note">167 tests from the course lab: 4 temperatures × 7 n-alkanes (n-C5 to n-C12) × about 6 dilution ratios.</p></div>
     <div id="asph-genbox" hidden>${rangeCtl("ga-noise", "Measurement noise (%)", 0, 20, 1, 5)}${rangeCtl("ga-out", "Planted outliers (% of tests)", 0, 20, 1, 6)}${rangeCtl("ga-fac", "Outlier size (× true value)", 1.3, 8, .1, 2)}
       <details class="more"><summary>More settings</summary>${rangeCtl("ga-ndr", "Dilution ratios per curve", 3, 20, 1, 6)}${rangeCtl("ga-seed", "Random seed", 0, 50, 1, 0)}</details>
       <p class="hint">Empirical model fitted to the clean course tests; the column is_outlier is the answer key.</p></div>`
  : `<div id="vfm-coursebox"><p class="note">Your field data: 2,100 well tests with choke, BHP, WHP, WHT, Psep and Tsep and the separator rates Qoil, Qwater and Qgas.</p></div>
     <div id="vfm-genbox" hidden style="display:none">${rangeCtl("gv-n", "Well tests", 50, 2000, 50, 600)}${rangeCtl("gv-noise", "Gauge noise (%)", 0, 5, .25, 1)}${rangeCtl("gv-wc", "Highest water cut", 0, .95, .05, .8)}
       <details class="more"><summary>More settings</summary>${rangeCtl("gv-gor", "GOR, middle of range (scf/STB)", 500, 8000, 250, 4000)}${rangeCtl("gv-depth", "Well depth (ft)", 4000, 14000, 500, 8000)}
       <label class="check"><input type="checkbox" id="gv-disc"> Only five water cuts (0 to 0.8), like the course data</label>${rangeCtl("gv-seed", "Random seed", 0, 50, 1, 0)}</details>
       <p class="hint">Inflow + tubing + Gilbert choke equations; reservoir pressure, PI, water cut and GOR are hidden from the model.</p></div>`); }
function regDataInit(k) { if (k === "asph") initSrc(k); if (k === "asph") { ["ga-out", "ga-ndr", "ga-seed", "ga-noise"].forEach(i => bind(i)); bind("ga-fac", v => v.toFixed(1)); }
  else { ["gv-n", "gv-gor", "gv-depth", "gv-seed"].forEach(i => bind(i)); bind("gv-noise", v => v.toFixed(2)); bind("gv-wc", v => v.toFixed(2)); } }
function regDataRun(k) { const s = S[k] = S[k] || {}; s.gen = null;
  if (k === "asph" && isGen(k)) { if (k === "asph") { s.gen = { k: "asph", seed: +$("ga-seed").value, noise: +$("ga-noise").value / 100, out: +$("ga-out").value / 100, fac: +$("ga-fac").value, ndr: +$("ga-ndr").value }; s.raw = genAsph(s.gen); }
    else { s.gen = { k: "vfm", seed: +$("gv-seed").value, n: +$("gv-n").value, noise: +$("gv-noise").value / 100, wcmax: +$("gv-wc").value, gor: +$("gv-gor").value, depth: +$("gv-depth").value, disc: $("gv-disc").checked }; s.raw = genVfm(s.gen); s.raw.forEach(r => r.case_id = r.test_id); } }
  else s.raw = (k === "asph" ? D.asph : D.vfm).map(r => ({ ...r }));
  if (k === "asph") s.raw.forEach(r => { r.series = `${r.temperature_K}K C${r.alkane_carbon_number}`; if (r.suspect === undefined) r.suspect = "normal"; }); }
function regDataRender(k) { const s = S[k], R = s.raw;
  if (k === "asph") { const ser = new Set(R.map(r => r.series)).size, T = [...new Set(R.map(r => r.temperature_K))].sort((a, b) => a - b);
    $("asph-met0").innerHTML = met(R.length, "tests") + met(ser, "curves (series)") + met(T.join(", "), "temperatures (K)") + met([...new Set(R.map(r => r.alkane_carbon_number))].length, "n-alkanes") + met(`${Math.min(...R.map(r => r.asphaltene_wt_pct)).toFixed(1)} – ${Math.max(...R.map(r => r.asphaltene_wt_pct)).toFixed(1)}`, "wt% range");
    $("asph-mean0").innerHTML = meaning("What you have", [`Each row is one lab test: crude oil diluted with an n-alkane at a temperature, and the weight percent of asphaltene that precipitated.`,
      `Inputs: temperature, n-alkane carbon number and dilution ratio (mL alkane per g oil). Target: <b>asphaltene_wt_pct</b>.`, s.gen ? ["w", `Generated data: ${R.filter(r => r.is_outlier).length} outliers were planted. Can you find them in step 2?`] : `Look at the curves on the right: most rise smoothly and level off. Do any look out of place?`]);
    const Ts = [293, 308, 323, 332], traces = [];
    [...new Set(R.map(r => r.series))].forEach(se => { const g = R.filter(r => r.series === se).sort((a, b) => a.dilution_ratio - b.dilution_ratio), ti = Ts.indexOf(g[0].temperature_K);
      traces.push({ x: g.map(r => r.dilution_ratio), y: g.map(r => r.asphaltene_wt_pct), mode: "lines+markers", type: "scatter", name: `${g[0].temperature_K} K`, legendgroup: g[0].temperature_K, showlegend: !traces.some(t => t.legendgroup === g[0].temperature_K), marker: { size: 5, color: CCOL[ti] }, line: { width: 1.3, color: CCOL[ti] }, hovertemplate: `${se}<br>DR %{x}<br>%{y} wt%<extra></extra>` }); });
    Plotly.react("asph-curves", traces, layout({ xaxis: { title: { text: "dilution ratio (mL/g)" } }, yaxis: { title: { text: "asphaltene precipitated (wt%)" } } }), CFG);
    const last = {}; R.forEach(r => { const key = r.series; if (!last[key] || r.dilution_ratio > last[key].dilution_ratio) last[key] = r; });
    Plotly.react("asph-cn", Ts.map((T, i) => { const g = Object.values(last).filter(r => r.temperature_K === T).sort((a, b) => a.alkane_carbon_number - b.alkane_carbon_number); return { x: g.map(r => r.alkane_carbon_number), y: g.map(r => r.asphaltene_wt_pct), name: T + " K", mode: "lines+markers", type: "scatter", marker: { color: CCOL[i] } }; }),
      layout({ xaxis: { title: { text: "n-alkane carbon number" }, dtick: 1 }, yaxis: { title: { text: "wt% at the highest dilution" } } }), CFG); }
  else { const tests = new Set(R.map(r => r.case_id)).size, liq = R.map(r => r.oil_rate_stb_d + r.water_rate_bbl_d);
    $("vfm-met0").innerHTML = met(R.length.toLocaleString(), "rows") + met([...new Set(R.map(r => r.choke_in))].length, "choke sizes") + met(Math.round(median(liq)).toLocaleString(), "median liquid rate (bbl/d)") + met(Math.round(Math.max(...liq)).toLocaleString(), "highest liquid rate");
    $("vfm-mean0").innerHTML = meaning("What you have", [`Each row is a well test: six gauge readings (choke, BHP, WHP, WHT, separator pressure and temperature) and the oil, water and gas rates the test separator measured.`,
      `A virtual flow meter learns rates from gauges so the well can be metered every minute, not once a month.`, `Count the rows, then ask in step 2 how many of them are really different tests.`]);
    Plotly.react("vfm-liq", [{ x: R.map(r => r.choke_in), y: liq, mode: "markers", type: "scatter", marker: { size: 5, color: R.map(r => r.water_cut ?? r.water_rate_bbl_d / Math.max(1, r.oil_rate_stb_d + r.water_rate_bbl_d)), colorscale: "Viridis", colorbar: { title: { text: "water cut" }, thickness: 10 } }, hovertemplate: "choke %{x} in<br>%{y:.0f} bbl/d<extra></extra>" }],
      layout({ xaxis: { title: { text: "choke (in)" } }, yaxis: { title: { text: "liquid rate (bbl/d)" } } }), CFG);
    Plotly.react("vfm-whp", [{ x: R.map(r => r.whp_psi), y: R.map(r => r.oil_rate_stb_d), mode: "markers", type: "scatter", marker: { size: 5, color: R.map(r => r.choke_in), colorscale: "Portland", colorbar: { title: { text: "choke" }, thickness: 10 } } }],
      layout({ xaxis: { title: { text: "wellhead pressure (psi)" } }, yaxis: { title: { text: "oil rate (STB/d)" } } }), CFG); } }
function genPyReg(o) { if (o.k === "vfm") return `from aiml.generators import make_vfm_data
df = make_vfm_data(n_tests=${o.n}, water_cut=(0.0, ${py(o.wcmax)}), gor_scf_stb=(${Math.round(.6 * o.gor)}, ${Math.round(1.4 * o.gor)}), depth_ft=${o.depth},
                   noise=${py(o.noise)},${o.disc ? " water_cut_levels=(0, 0.2, 0.4, 0.6, 0.8)," : ""} seed=${o.seed})
df["case_id"] = df.test_id`;
  return `from aiml.generators import make_asphaltene_data
df = make_asphaltene_data(dilution_ratios=tuple(np.linspace(3.6, 24.4, ${o.ndr})), noise=${py(o.noise)}, outliers=${py(o.out)},
                          outlier_factor=(${py(.8 * o.fac)}, ${py(1.2 * o.fac)}), seed=${o.seed})
df["series"] = df.temperature_K.astype(str) + "K_C" + df.alkane_carbon_number.astype(str)
df["suspect"] = False`; }

// ---------- python, cumulative per step ----------
function pyReg(k, upto) { const eda = upto === "eda"; if (eda) upto = 0; const s = S[k], c = REGCFG[k], out = [`# ${k === "asph" ? "Asphaltene precipitation" : "Virtual flow metering"}: the workflow up to step ${eda ? 2 : upto === 0 ? 1 : upto + 2}  (Part I, ${k === "asph" ? "Lab 1" : "Lab 3"})`, KIT1, PYIMP];
  out.push(`\n# ---- Step 1 · get the data ----`, s.gen ? genPyReg(s.gen) : k === "asph" ? `from aiml.data import load_asphaltene\ndf = load_asphaltene()                 # adds 'series' and 'suspect'` : `from aiml.data import load_vfm\ndf = load_vfm()                        # adds 'case_id': the copies of one well test share it`,
    `TARGETS = ${py(c.targets)}`, `print(df.shape); print(df.head())`);
  if ((eda || upto >= 1) && s.eda) out.push(pyEda(k));
  if (upto >= 1 && s.dec) { const d = s.dec; out.push(`\n# ---- Step 3 · clean ----`, `from aiml.prep import detect_outliers_iqr, outlier_table`, `print(outlier_table(df, threshold=${py(d.k)}))`, `drop = set()`);
    if (d.suspect) out.push(`drop |= set(df.index[df.suspect])                  # the 17 tests at 332 K with n-C6 to n-C8`);
    if (d.iqr) out.push(`for t in TARGETS:\n    drop |= set(detect_outliers_iqr(df, t, threshold=${py(d.k)})[2])`);
    out.push(`clean = df.drop(index=list(drop))`);
    if (d.dedupe) out.push(`clean = clean.drop_duplicates(subset="case_id")       # one row per well test`);
    out.push(`clean = clean.reset_index(drop=True)`);
    if (d.logdr) out.push(`clean["log_dilution"] = np.log(clean.dilution_ratio)`);
    if (d.invt) out.push(`clean["inv_temperature"] = 1000 / clean.temperature_K`);
    out.push(`print(len(df), "rows ->", len(clean), "after cleaning")`); }
  if (upto >= 2 && s.split) { const p = s.split; out.push(`\n# ---- Step 4 · split ----`);
    if (p.method === "random") out.push(`_, te = train_test_split(np.arange(len(clean)), test_size=${py(p.share)}, random_state=${p.seed})\ntest = np.isin(np.arange(len(clean)), te)`);
    else if (p.method === "group") out.push(`groups = clean[${py(k === "asph" ? "series" : "case_id")}]\n_, te = next(GroupShuffleSplit(n_splits=1, test_size=${py(p.share)}, random_state=${p.seed}).split(clean, groups=groups))\ntest = np.isin(np.arange(len(clean)), te)`);
    else out.push(`test = (clean[${py(p.col)}] == ${p.val}).values      # hold out one value`);
    out.push(`train = clean[~test]`, `print(len(train), "train rows,", test.sum(), "test rows")`);
    if (k === "vfm") out.push(`gauges_and_rates = ["choke_in", "bhp_psi", "whp_psi", "wht_F", "tsep_F"] + TARGETS\nleak = clean[test].merge(train[gauges_and_rates].drop_duplicates(), on=gauges_and_rates).shape[0]\nprint("test rows whose well test is also in training:", leak)`); }
  if (upto >= 3 && s.model) { const m = s.model; out.push(`\n# ---- Step 5 · compare with cross-validation, then train the final model ----`, `from aiml.models import regressors`, `INPUTS = ${py(m.inputs)}`,
      s.split.method === "group" ? `cv, cv_groups = GroupKFold(${m.K}), train[${py(k === "asph" ? "series" : "case_id")}]          # k = ${m.K} folds` : `cv, cv_groups = KFold(${m.K}, shuffle=True, random_state=0), None          # k = ${m.K} folds`,
      `for name in ${py(m.compared.map(a => AIML[a]))}:\n    sc = cross_val_score(regressors()[name], train[INPUTS], train[TARGETS[0]], cv=cv, groups=cv_groups, scoring="r2")\n    print(f"{name:<28} CV R2 {sc.mean():.3f} ± {sc.std():.3f}   folds: {np.round(sc, 3)}")`,
      ...(m.sel && m.cv.some(r => r.alg === m.sel) ? [`# results of one algorithm: ${ALGN[m.sel]}, every training row predicted by the model trained without it
from sklearn.model_selection import cross_val_predict
from aiml.evaluate import report
oof = cross_val_predict(regressors()[${py(AIML[m.sel])}], train[INPUTS], train[TARGETS[0]], cv=cv, groups=cv_groups)
report(train[TARGETS[0]], oof, ${py(AIML[m.sel] + " out-of-fold")})
plt.scatter(train[TARGETS[0]], oof, s=8, alpha=0.5); lim = [train[TARGETS[0]].min(), train[TARGETS[0]].max()]
plt.plot(lim, lim, "k--"); plt.xlabel("measured"); plt.ylabel("out-of-fold prediction"); plt.title(${py(AIML[m.sel])}); plt.show()`] : []),
      `final = ${PYREG[m.alg](m.h, 0)}`, `models = {t: clone(final).fit(train[INPUTS], train[t]) for t in TARGETS}`); }
  if (upto >= 4 && s.ev) { out.push(`\n# ---- Step 6 · evaluate once on the test set ----`, `from aiml.evaluate import report, crossplot, importance`, `for t in TARGETS:\n    report(train[t], models[t].predict(train[INPUTS]), t + " train")\n    report(clean[test][t], models[t].predict(clean[test][INPUTS]), t + " test")`,
      `fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))\nt = TARGETS[0]\ncrossplot(train[t], models[t].predict(train[INPUTS]), clean[test][t], models[t].predict(clean[test][INPUTS]), t, ax=ax[0])\nimportance(models[t], clean[test][INPUTS], clean[test][t], ax=ax[1]); plt.tight_layout(); plt.show()`,
      `${pyShapNote}\nimport shap\nbg = train[INPUTS].sample(min(50, len(train)), random_state=0)\nX_explain = clean[test][INPUTS].iloc[:150]\nt = ${py(s.ev.shapT || c.targets[0])}\nexplainer = shap.Explainer(lambda X: models[t].predict(pd.DataFrame(X, columns=INPUTS)), bg)\nsv = explainer(X_explain)\nshap.plots.beeswarm(sv, max_display=len(INPUTS))\nshap.plots.bar(sv, max_display=len(INPUTS))\ntop = INPUTS[int(np.abs(sv.values).mean(0).argmax())]\nshap.plots.scatter(sv[:, top], color=sv)`); }
  if (upto >= 5 && s.use) { out.push(`\n# ---- Step 7 · use the model ----`);
    if (k === "asph") { const u = s.use; out.push(`dr = np.linspace(${u.dr0}, ${u.dr1}, 50)\nnew = pd.DataFrame({"temperature_K": ${u.T}, "alkane_carbon_number": ${u.C}, "dilution_ratio": dr})\nnew["log_dilution"] = np.log(new.dilution_ratio); new["inv_temperature"] = 1000 / new.temperature_K\nnew["predicted_wt_pct"] = models[TARGETS[0]].predict(new[INPUTS])\nplt.plot(new.dilution_ratio, new.predicted_wt_pct); plt.xlabel("dilution ratio (mL/g)"); plt.ylabel("predicted wt%"); plt.title("${u.T} K, n-C${u.C}"); plt.show()\nprint(new.iloc[::7].round(3))`); }
    else { const g = s.use.g; out.push(`reading = pd.DataFrame([${"{"}${VFM_F.map(f => `"${f}": ${py(g[f])}`).join(", ")}${"}"}])\nprint({t: round(float(m.predict(reading[INPUTS])[0]), 2) for t, m in models.items()})\n# the meter against the test separator, on wells it never saw\nfor t, c in [("oil_rate_stb_d", "tab:orange"), ("water_rate_bbl_d", "tab:blue")]:\n    tw = clean[test].sort_values(t).reset_index(drop=True)\n    plt.plot(tw.index, tw[t], "o", color=c, alpha=0.4, ms=3, label=t + " separator")\n    plt.plot(tw.index, models[t].predict(tw[INPUTS]).clip(0), "-", color=c, label=t + " meter")\nplt.xlabel("test wells, sorted by measured rate"); plt.ylabel("rate"); plt.legend(); plt.show()`); } }
  return out.join("\n"); }
function pyNote(upto) { return `The Python script for steps 1 to ${upto + 1}, with the settings on this page. ${upto === 6 ? "This is the complete workflow." : "Each step adds to the script of the step before."} Browser models are small versions of the scikit-learn ones, so numbers are close, not identical.`; }

// ---------- step 2 · explore and clean ----------
function regCleanControls(k) { return (k === "asph"
  ? decision("asph-d-suspect", "Remove the 17 suspect tests", "332 K with n-C6, C7 and C8: about ten times higher than every other series. Only for the course data.", false)
    + decision("asph-d-iqr", "Remove IQR outliers of the target", "Drops every test outside the IQR fences below.", false)
    + decision("asph-d-logdr", "Add log(dilution ratio) as an input", "Precipitation grows roughly with the log of dilution; a straight-line model cannot bend by itself.", false)
    + decision("asph-d-invt", "Add 1000/temperature as an input", "The form temperature takes in solubility equations.", false)
  : decision("vfm-d-dedupe", "Keep one row per well test", "Drops the copies of each test at other separator pressures (the rates do not change with Psep).", false)
    + decision("vfm-d-iqr", "Remove IQR outliers of the rates", "", false))
  + rangeCtl(`${k}-d-k`, "IQR multiplier k", 1, 3, .1, 1.5, "Flag below Q1 − k·IQR or above Q3 + k·IQR."); }
function regCleanInit(k) { bind(`${k}-d-k`, v => v.toFixed(1)); }
function regCleanOnEnter(k) { if (k === "asph" && S.asph && $("asph-d-suspect")) { const g = !!S.asph.gen; $("asph-d-suspect").disabled = g; if (g) $("asph-d-suspect").checked = false; } }
function regCleanRun(k) { const s = S[k], c = REGCFG[k], R = s.raw, kk = +$(`${k}-d-k`).value;
  const d = { k: kk, suspect: k === "asph" && $("asph-d-suspect").checked && !s.gen, iqr: $(`${k}-d-iqr`).checked, logdr: k === "asph" && $("asph-d-logdr").checked, invt: k === "asph" && $("asph-d-invt").checked, dedupe: k === "vfm" && $("vfm-d-dedupe").checked };
  const fences = {}, flag = R.map(() => false); c.targets.forEach(t => { const [lo, hi] = iqr(R.map(r => r[t]), kk); fences[t] = [lo, hi]; R.forEach((r, i) => { if (r[t] < lo || r[t] > hi) flag[i] = true; }); });
  const why = R.map((r, i) => d.suspect && r.suspect === "suspect" ? "suspect series" : d.iqr && flag[i] ? "IQR outlier" : "");
  let keep = R.filter((_, i) => !why[i]), dup = 0;
  if (d.dedupe) { const seen = new Set(); keep = keep.filter(r => { if (seen.has(r.case_id)) { dup++; return false; } seen.add(r.case_id); return true; }); }
  s.dec = d; s.flag = flag; s.fences = fences; s.removed = R.map((r, i) => ({ ...r, removed_because: why[i] })).filter(r => r.removed_because); s.clean = keep; s.dupDropped = dup;
  s.feats = [...c.base, ...(d.logdr ? ["log_dilution"] : []), ...(d.invt ? ["inv_temperature"] : [])]; }
function regCleanRender(k) { const s = S[k], c = REGCFG[k], R = s.raw, d = s.dec, t0 = c.targets[0], nflag = s.flag.filter(Boolean).length;
  const tests = new Set(R.map(r => r.case_id ?? r.series + r.dilution_ratio)).size;
  $(`${k}-met2`).innerHTML = met(R.length, "rows before") + met(nflag, `IQR flags (k = ${d.k})`, nflag ? "warn" : "") + (k === "vfm" ? met(tests.toLocaleString(), "different well tests", tests < R.length ? "warn" : "") : met(R.filter(r => r.suspect === "suspect").length, "suspect tests", "warn"))
    + met(R.length - s.clean.length, "rows removed") + met(s.clean.length, "rows kept", "train") + met(s.feats.length, "inputs available");
  const items = [];
  if (k === "asph") { const bySer = {}; R.forEach((r, i) => { if (s.flag[i]) bySer[r.series] = (bySer[r.series] || 0) + 1; });
    const top = Object.entries(bySer).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([a, b]) => `${a} (${b})`).join(", ");
    items.push(nflag ? ["w", `The IQR rule flags ${nflag} tests${top ? `, mostly in ${top}` : ""}. ${s.gen ? `The generator planted ${R.filter(r => r.is_outlier).length}; it caught ${R.filter((r, i) => r.is_outlier && s.flag[i]).length}.` : "They are not random spikes: they are whole curves, which points to a different oil, unit or procedure rather than noise."}`] : "No test is outside the IQR fences.");
    if (!s.gen) items.push(d.suspect ? ["g", "The 17 suspect tests are removed. Compare step 4 with and without them: it changes which model wins."] : ["w", "The 17 suspect tests are still in. Flexible models will chase them; straight-line models will fail."]);
    items.push(d.logdr ? ["g", "log_dilution is now an input. Linear models gain the most from it."] : "Try adding log(dilution): the curves bend like a logarithm."); }
  else { items.push(["w", `${R.length.toLocaleString()} rows hold only ${tests.toLocaleString()} different well tests. Each test is repeated for ${Math.round(R.length / tests)} separator pressures with the same rates, so separator pressure has no effect (left chart).`]);
    items.push(d.dedupe ? ["g", `Kept one row per test (${s.dupDropped.toLocaleString()} copies dropped). A random split can no longer put copies on both sides.`] : ["w", "Copies are kept. In step 3, split by well test, or a random split will test the model on rows it has already seen."]);
    items.push(nflag ? `IQR flags ${nflag} rows: wide ranges by design (ten chokes, water cut up to 0.8), not errors. Removing them would delete real high-rate tests.` : "No rate is outside the IQR fences: wide ranges here are by design, not errors."); }
  $(`${k}-mean2`).innerHTML = meaning("What the check found", items);
  if (k === "asph") { const rem = new Set(s.removed.map(r => r.series + "|" + r.dilution_ratio)), kept = R.filter(r => !rem.has(r.series + "|" + r.dilution_ratio)), gone = R.filter(r => rem.has(r.series + "|" + r.dilution_ratio)), fl = R.filter((_, i) => s.flag[i]);
    Plotly.react("asph-clean-sc", [{ x: kept.map(r => r.dilution_ratio), y: kept.map(r => r.asphaltene_wt_pct), mode: "markers", type: "scatter", name: "kept", marker: { size: 6, color: TR() } },
      { x: gone.map(r => r.dilution_ratio), y: gone.map(r => r.asphaltene_wt_pct), mode: "markers", type: "scatter", name: "removed", marker: { size: 8, symbol: "x", color: css("--bad") } },
      { x: fl.map(r => r.dilution_ratio), y: fl.map(r => r.asphaltene_wt_pct), mode: "markers", type: "scatter", name: "IQR flag", marker: { size: 13, color: "rgba(0,0,0,0)", line: { color: TE(), width: 2 } } }],
      layout({ xaxis: { title: { text: "dilution ratio (mL/g)" } }, yaxis: { title: { text: "wt%" } } }), CFG); }
  else { const ids = [...new Set(R.map(r => r.case_id))].filter((_, i) => i % Math.max(1, Math.floor(tests / 8)) === 0).slice(0, 8);
    Plotly.react("vfm-clean-sc", ids.map((id, i) => { const g = R.filter(r => r.case_id === id).sort((a, b) => a.psep_psi - b.psep_psi); return { x: g.map(r => r.psep_psi), y: g.map(r => r.oil_rate_stb_d), mode: "lines+markers", type: "scatter", name: `test ${id}`, showlegend: false, line: { color: CCOL[i % 10] } }; }),
      layout({ xaxis: { title: { text: "separator pressure (psi)" } }, yaxis: { title: { text: "oil rate (STB/d)" } } }), CFG); }
  const [lo, hi] = s.fences[t0], v = R.map(r => r[t0]), mn = Math.min(...v), mx = Math.max(...v), pad = .1 * (mx - mn);
  Plotly.react(`${k}-clean-hist`, [{ x: v, type: "histogram", nbinsx: 40, marker: { color: TR() }, name: t0 }], layout({ bargap: .05, xaxis: { title: { text: t0 } }, yaxis: { title: { text: "rows" } },
    shapes: [lo, hi].filter(b => b >= mn - pad && b <= mx + pad).map(b => ({ type: "line", x0: b, x1: b, yref: "paper", y0: 0, y1: 1, line: { color: TE(), dash: "dash", width: 2 } })) }), CFG);
  $(`${k}-clean-hist-read`).textContent = `${t0}: dashed lines are the IQR fences (k = ${d.k}) when they fall inside the data range.`;
  $(`${k}-iqrtab`).innerHTML = tableHTML([...c.base, ...c.targets].map(col => { const vv = R.map(r => r[col]), [l, h] = iqr(vv, d.k), n = vv.filter(x => x < l || x > h).length; return { column: col, min: Math.min(...vv), median: +median(vv).toFixed(3), max: Math.max(...vv), "IQR lower": +l.toFixed(3), "IQR upper": +h.toFixed(3), flagged: n }; }), 20); }

// ---------- step 3 · split ----------
function regSplitControls(k) { const c = REGCFG[k]; return `<div class="f"><label for="${k}-sp-m">How to choose the test rows</label><select id="${k}-sp-m"><option value="random">Random rows</option><option value="group">Whole ${c.groupPlural}</option><option value="hold">Hold out one value</option></select><p class="hint" id="${k}-sp-note"></p></div>
  <div id="${k}-sp-share">${rangeCtl(`${k}-sp-frac`, "Share for testing", .1, .4, .05, .2)}</div>
  <div id="${k}-sp-hold" hidden><div class="f"><label for="${k}-sp-col">Hold out</label><select id="${k}-sp-col">${c.hold.map(h => `<option>${h}</option>`).join("")}</select></div><div class="f"><label for="${k}-sp-val">Value</label><select id="${k}-sp-val"></select></div></div>
  <details class="more"><summary>Random seed</summary>${rangeCtl(`${k}-sp-seed`, "Seed", 0, 30, 1, 0)}</details>`; }
function regSplitInit(k) { bind(`${k}-sp-frac`, v => Math.round(v * 100) + " %"); bind(`${k}-sp-seed`);
  const note = () => { const m = $(`${k}-sp-m`).value; $(`${k}-sp-share`).hidden = m === "hold"; $(`${k}-sp-hold`).hidden = m !== "hold";
    $(`${k}-sp-note`).textContent = m === "random" ? (k === "vfm" ? "Easy test: rows are shuffled, so copies of one well test can land on both sides." : "Easy test: points from the same curve land on both sides, so the model only fills gaps in curves it has seen.") : m === "group" ? `Honest test: ${REGCFG[k].groupWord}.` : "Hardest test: everything at one value is unseen. At the edge of the range this is extrapolation."; };
  $(`${k}-sp-m`).addEventListener("change", note); $(`${k}-sp-col`).addEventListener("change", () => regHoldVals(k)); note(); }
function regHoldVals(k) { const s = S[k]; if (!s || !s.clean) return; const c = $(`${k}-sp-col`).value, v = [...new Set(s.clean.map(r => r[c]))].sort((a, b) => a - b); $(`${k}-sp-val`).innerHTML = v.map(x => `<option>${x}</option>`).join(""); if (k === "vfm") $(`${k}-sp-val`).selectedIndex = Math.floor(v.length / 2); }
function regSplitRun(k) { const s = S[k], R = s.clean, m = $(`${k}-sp-m`).value, share = +$(`${k}-sp-frac`).value, seed = +$(`${k}-sp-seed`).value, rng = seeded(seed * 31 + 5), c = REGCFG[k]; let test;
  if (m === "random") { const ix = shuffleIdx(R.length, rng), te = new Set(ix.slice(0, Math.round(share * R.length))); test = R.map((_, i) => te.has(i)); }
  else if (m === "group") { const g = [...new Set(R.map(c.group))], ix = shuffleIdx(g.length, rng), held = new Set(ix.slice(0, Math.max(1, Math.round(share * g.length))).map(i => g[i])); test = R.map(r => held.has(c.group(r))); }
  else { const col = $(`${k}-sp-col`).value, val = +$(`${k}-sp-val`).value; test = R.map(r => r[col] === val); }
  s.split = { method: m, share, seed, test, col: $(`${k}-sp-col`).value, val: +$(`${k}-sp-val`).value };
  if (k === "vfm") { const key = r => [r.choke_in, r.bhp_psi, r.whp_psi, r.wht_F, r.tsep_F, r.oil_rate_stb_d].join("|"), tr = new Set(R.filter((_, i) => !test[i]).map(key)); s.split.leak = R.filter((r, i) => test[i] && tr.has(key(r))).length; }
  else { const trs = new Set(R.filter((_, i) => !test[i]).map(c.group)); s.split.shared = new Set(R.filter((_, i) => test[i]).map(c.group).filter(g => trs.has(g))).size; s.split.testSeries = new Set(R.filter((_, i) => test[i]).map(c.group)).size; } }
function regSplitRender(k) { const s = S[k], R = s.clean, p = s.split, t0 = REGCFG[k].targets[0], nte = p.test.filter(Boolean).length;
  const tr = R.filter((_, i) => !p.test[i]), te = R.filter((_, i) => p.test[i]);
  $(`${k}-met3`).innerHTML = met(tr.length.toLocaleString(), "training rows", "train") + met(nte.toLocaleString(), "test rows", "test")
    + (k === "vfm" ? met(p.leak.toLocaleString(), "test rows with a copy in training", p.leak ? "bad" : "good") : met(`${p.shared} of ${p.testSeries}`, "test curves also seen in training", p.shared ? "warn" : "good"));
  const items = [];
  if (!nte) items.push(["b", "The test set is empty. Choose another value or method."]);
  if (k === "vfm") items.push(p.leak ? ["b", `${p.leak.toLocaleString()} test rows (${Math.round(100 * p.leak / Math.max(1, nte))} %) are copies of training rows. The test score will be too optimistic: this is leakage. Split by well test, or keep one row per test in step 2.`] : ["g", "No test row has a copy in training. The test score will be honest."]);
  else items.push(p.shared ? ["w", `Points from ${p.shared} curves are on both sides. The model will mostly interpolate along curves it has already seen.`] : ["g", "Every test curve is completely unseen. This is the honest test for a new temperature or a new precipitant."]);
  if (p.method === "hold") { const all = R.map(r => r[p.col]), edge = p.val === Math.min(...all) || p.val === Math.max(...all); items.push(edge ? ["w", `${p.col} = ${p.val} is at the edge of the data: the model must extrapolate. Tree models cannot predict beyond what they have seen.`] : `${p.col} = ${p.val} is inside the range: the model interpolates between neighbours.`); }
  $(`${k}-mean3`).innerHTML = meaning("What this split tests", items);
  const xcol = k === "asph" ? "dilution_ratio" : "whp_psi";
  Plotly.react(`${k}-split-sc`, [{ x: tr.map(r => r[xcol]), y: tr.map(r => r[t0]), mode: "markers", type: "scatter", name: "train", marker: { size: 5, color: TR(), opacity: .6 } }, { x: te.map(r => r[xcol]), y: te.map(r => r[t0]), mode: "markers", type: "scatter", name: "test", marker: { size: 7, color: TE() } }],
    layout({ xaxis: { title: { text: xcol } }, yaxis: { title: { text: t0 } } }), CFG);
  Plotly.react(`${k}-split-h`, [{ x: tr.map(r => r[t0]), type: "histogram", name: "train", histnorm: "probability", opacity: .6, marker: { color: TR() } }, { x: te.map(r => r[t0]), type: "histogram", name: "test", histnorm: "probability", opacity: .7, marker: { color: TE() } }],
    layout({ barmode: "overlay", xaxis: { title: { text: t0 } }, yaxis: { title: { text: "share of rows" } } }), CFG); }

// ---------- step 4 · compare and train ----------
function regTrainControls(k) { return `<div class="f"><span class="lab-t">Inputs</span><div class="checks full" id="${k}-tr-in"></div></div>
  ${rangeCtl(`${k}-tr-k`, "Number of folds k (k-fold cross-validation)", 3, 10, 1, 5, "Each algorithm is trained k times, each time scored on a different 1/k of the training rows.")}
  <div class="f"><span class="lab-t">Algorithms to compare (k-fold cross-validation on training rows)</span><div class="checks full" id="${k}-tr-cmp">${Object.entries(ALGN).map(([a, n]) => `<label class="check"><input type="checkbox" value="${a}" ${a !== "nn" || k === "asph" ? "checked" : ""}> ${n}</label>`).join("")}</div></div>
  <div class="f"><label for="${k}-tr-alg">Final model</label><select id="${k}-tr-alg"><option value="best">The best one from the comparison</option>${Object.entries(ALGN).map(([a, n]) => `<option value="${a}">${n}</option>`).join("")}</select></div>
  <div id="${k}-tr-hp"></div>`; }
function regTrainInit(k) { bind(`${k}-tr-k`); algWire(k, regAlgDraw, `${k}-tr-alg`); $(`${k}-tr-alg`).addEventListener("change", () => { const a = $(`${k}-tr-alg`).value; $(`${k}-tr-hp`).innerHTML = a === "best" ? `<p class="hint">Uses the comparison settings. Pick a model by name to set its own settings.</p>` : `<details class="more" open><summary>Settings for ${ALGN[a]}</summary>${hpHTML(`${k}-h`, a, REGCFG[k].big)}</details>`; if (a !== "best") hpBind(`${k}-h`, a); }); $(`${k}-tr-alg`).dispatchEvent(new Event("change")); }
function regTrainOnEnter(k) { const s = S[k]; if (!s || !s.feats) return; const box = $(`${k}-tr-in`), was = [...box.querySelectorAll("input")].map(e => [e.value, e.checked]), prev = Object.fromEntries(was);
  box.innerHTML = s.feats.map(f => `<label class="check"><input type="checkbox" value="${f}" ${prev[f] === false ? "" : "checked"}> ${f}</label>`).join(""); }
async function regTrainRun(k) { const s = S[k], c = REGCFG[k], R = s.clean, p = s.split, inputs = [...$(`${k}-tr-in`).querySelectorAll("input:checked")].map(e => e.value);
  if (!inputs.length) throw new Error("choose at least one input");
  const trR = R.filter((_, i) => !p.test[i]), F = fx(k, inputs), X = trR.map(F), y0 = trR.map(r => r[c.targets[0]]), cmp = [...$(`${k}-tr-cmp`).querySelectorAll("input:checked")].map(e => e.value), K = +$(`${k}-tr-k`).value;
  const folds = kfold(trR.length, K, seeded(11), p.method === "group" ? trR.map(c.group) : null), res = [];
  for (let a = 0; a < cmp.length; a++) { const alg = cmp[a], h = { ...HPDEF(alg), ...(c.big ? BIGFIX[alg] || {} : {}) }, sc = [], oof = new Array(X.length).fill(NaN), ta = performance.now();
    for (let f = 0; f < K; f++) { const ti = [], vi = []; folds.forEach((q, i) => (q === f ? vi : ti).push(i)); if (!vi.length) continue;
      const m = makeModel(alg, h, "reg", 0).fit(ti.map(i => X[i]), ti.map(i => y0[i])), pv = m.predict(vi.map(i => X[i])); vi.forEach((i, j) => oof[i] = pv[j]); sc.push(regMetrics(vi.map(i => y0[i]), pv).R2); await prog(k, 4, (a * K + f + 1) / (cmp.length * K + 3)); }
    res.push({ alg, mean: mean(sc), sd: sd(sc), folds: sc, h, oof, M: regMetrics(y0, oof), sec: (performance.now() - ta) / 1000 }); }
  res.sort((a, b) => b.mean - a.mean);
  let alg = $(`${k}-tr-alg`).value, h; if (alg === "best") { if (!res.length) throw new Error("tick at least one algorithm to compare, or pick a final model"); alg = res[0].alg; h = res[0].h; } else h = hpRead(`${k}-h`, alg);
  const t0 = performance.now(), models = {}; for (let i = 0; i < c.targets.length; i++) { const t = c.targets[i]; models[t] = makeModel(alg, h, "reg", 0).fit(X, trR.map(r => r[t])); await prog(k, 4, (cmp.length * K + i + 1) / (cmp.length * K + 3)); }
  const brng = seeded(21), bg = shuffleIdx(X.length, brng).slice(0, Math.min(40, X.length)).map(i => X[i]);
  s.model = { alg, h, inputs, models, sec: (performance.now() - t0) / 1000, cv: res, compared: cmp, F, best: res[0], K, fold: folds, y0, bg, sel: res.some(r => r.alg === alg) ? alg : res[0] && res[0].alg }; }
function regTrainRender(k) { const s = S[k], m = s.model, best = m.best, c = REGCFG[k];
  $(`${k}-met4`).innerHTML = (best ? met(best.mean.toFixed(3), `best CV R² (${ALGN[best.alg]})`, "train") : "") + met(ALGN[m.alg], "final model") + met(m.inputs.length, "inputs") + met(c.targets.length, c.targets.length > 1 ? "models (one per phase)" : "model") + met(msec(m.sec), "training time");
  const items = []; if (best) { items.push(`${m.K}-fold cross-validation trains each algorithm ${m.K} times, each time on ${Math.round(100 * (m.K - 1) / m.K)} % of the training rows, and scores it on the other ${Math.round(100 / m.K)} %. Every training row is scored once. ${s.split.method === "group" ? `Folds keep whole ${c.groupPlural} together, like the test set.` : ""} The test set is still untouched.`);
    const worst = m.cv[m.cv.length - 1]; items.push(`Best: <b>${ALGN[best.alg]}</b> (R² ${best.mean.toFixed(3)} ± ${best.sd.toFixed(3)}). Weakest: ${ALGN[worst.alg]} (${worst.mean.toFixed(3)}).`);
    const close = m.cv.filter(r => r !== best && best.mean - r.mean < Math.max(best.sd, .01)); if (close.length) items.push(["w", `${close.map(r => ALGN[r.alg]).join(", ")} ${close.length > 1 ? "are" : "is"} within one standard deviation of the best: the difference may be luck of the folds. Prefer the simpler model when scores tie.`]);
    if (m.alg !== best.alg) items.push(["w", `You chose ${ALGN[m.alg]} as the final model, not the best by cross-validation. That is fine if you have a reason (simpler, smoother, more physical).`]); }
  if (c.targets.length > 1) items.push(`The comparison uses the oil rate. The final model is trained three times: oil, water and gas.`);
  $(`${k}-mean4`).innerHTML = meaning("What the comparison says", items);
  const cv = [...m.cv].reverse(); $(`${k}-cv-card`).hidden = !cv.length;
  if (cv.length) Plotly.react(`${k}-cv`, [{ y: cv.map(r => ALGN[r.alg]), x: cv.map(r => Math.max(-0.5, r.mean)), error_x: { type: "data", array: cv.map(r => r.sd), color: MUTE() }, type: "bar", orientation: "h", marker: { color: cv.map(r => r.alg === m.alg ? TE() : TR()) }, text: cv.map(r => r.mean.toFixed(3)), textposition: "auto", hovertemplate: "%{y}: %{text}<extra></extra>" }],
    layout({ margin: { l: 170, r: 16, t: 8, b: 44 }, xaxis: { title: { text: `mean R² over ${m.K} folds (amber = final model)` }, range: [Math.min(0, ...cv.map(r => r.mean)) - .02, 1.02] } }), CFG);
  if (cv.length) { cvClick(k, `${k}-cv`, ALGN, regAlgDraw); foldDiagram(`${k}-kfold`, m.fold, m.K, s.split.method === "group" ? c.groupPlural : ""); regAlgDraw(k); }
  const nn = m.models[c.targets[0]].curve; $(`${k}-curve-card`).hidden = !nn; $(`${k}-cv-card`).classList.toggle("span", !nn);
  if (nn) Plotly.react(`${k}-curve`, [{ x: nn.map(q => q[0]), y: nn.map(q => q[1]), mode: "lines", type: "scatter", line: { color: TR(), width: 2.5 }, name: "training loss" }], layout({ xaxis: { title: { text: "epoch" } }, yaxis: { title: { text: "loss (scaled)" }, type: "log" } }), CFG); }

function regAlgDraw(k) { const s = S[k], m = s.model, c = REGCFG[k]; if (!m || !m.cv.length) return; const r = m.cv.find(q => q.alg === m.sel) || m.cv[0], t0 = c.targets[0], u = c.units[t0];
  algButtons(k, m.cv, r.alg, ALGN); foldLines(`${k}-cvfold`, m.cv, r.alg, ALGN, "R² of the fold");
  $(`${k}-algmet`).innerHTML = met(r.mean.toFixed(3), `CV R² (mean of ${r.folds.length})`, r2c(r.mean)) + met("± " + r.sd.toFixed(3), "spread between folds", r.sd > .05 ? "warn" : "") + met(fmt(r.M.MAE), `MAE (${u})`) + met(fmt(r.M.RMSE), `RMSE (${u})`) + met(r.M.MAPE.toFixed(1) + " %", "MAPE") + met(msec(r.sec), `time for ${r.folds.length} fits`);
  const bf = r.folds.indexOf(Math.max(...r.folds)), wf = r.folds.indexOf(Math.min(...r.folds));
  $(`${k}-algnote`).innerHTML = `<b>${ALGN[r.alg]}</b>: fold scores ${r.folds.map(v => v.toFixed(3)).join(", ")}. Best fold ${bf + 1}, weakest fold ${wf + 1}.${r.alg === m.alg ? " This is the final model." : ""}${c.targets.length > 1 ? " Scores are for the oil rate." : ""}`;
  $(`${k}-algfinal`).hidden = r.alg === m.alg;
  const lo = Math.min(...m.y0, ...r.oof), hi = Math.max(...m.y0, ...r.oof);
  Plotly.react(`${k}-algpar`, [{ x: m.y0, y: r.oof, mode: "markers", type: "scatter", name: ALGN[r.alg], marker: { size: 5, color: m.fold.map(f => CCOL[f % 10]), opacity: .7 }, text: m.fold.map(f => `fold ${f + 1}`), hovertemplate: "%{text}<br>measured %{x:.4g}<br>predicted %{y:.4g}<extra></extra>" },
    { x: [lo, hi], y: [lo, hi], mode: "lines", type: "scatter", name: "perfect", line: { color: MUTE(), dash: "dash" } }], layout({ showlegend: false, xaxis: { title: { text: `measured ${t0}` } }, yaxis: { title: { text: "out-of-fold prediction" } } }), CFG);
  $(`${k}-algpar-read`).textContent = `${ALGN[r.alg]}: every training row predicted by the model trained without its fold. Colour = fold.`;
  dcRefresh(`dc-${k}-4`); }

// ---------- step 5 · evaluate ----------
async function regEvalRun(k) { const s = S[k], c = REGCFG[k], R = s.clean, p = s.split, m = s.model, tr = R.filter((_, i) => !p.test[i]), te = R.filter((_, i) => p.test[i]);
  if (!te.length) throw new Error("the test set is empty: change the split in step 3"); const Xtr = tr.map(m.F), Xte = te.map(m.F), out = {};
  for (const t of c.targets) { const ptr = m.models[t].predict(Xtr), pte = m.models[t].predict(Xte); out[t] = { ptr, pte, mt: regMetrics(tr.map(r => r[t]), ptr), me: regMetrics(te.map(r => r[t]), pte) }; }
  await prog(k, 5, .3); const t0 = c.targets[0], imp = permImportance(m.models[t0], Xte, te.map(r => r[t0]), (y, q) => regMetrics(y, q).R2, seeded(9), m.alg === "rf" || m.alg === "nn" ? 2 : 3);
  const xi = shuffleIdx(Xte.length, seeded(17)).slice(0, Math.min(Xte.length, k === "vfm" ? 120 : 150)), Xs = xi.map(i => Xte[i]), MM = Xs.length * m.inputs.length > 400 ? 8 : 16;
  const fAll = rows => { const P = c.targets.map(t => m.models[t].predict(rows)); return rows.map((_, i) => P.map(q => q[i])); };
  const sh = await shapValues(fAll, Xs, m.bg, MM, 13, f => prog(k, 5, .6 + .4 * f));
  s.ev = { tr, te, out, imp: m.inputs.map((f, i) => [f, imp[i]]).sort((a, b) => b[1] - a[1]), shap: sh, shapX: Xs, shapT: (s.ev && s.ev.shapT) || c.targets[0] }; }
function regEvalRender(k) { const s = S[k], c = REGCFG[k], e = s.ev, m = s.model, t0 = c.targets[0], o = e.out[t0];
  $(`${k}-met5`).innerHTML = c.targets.length > 1 ? c.targets.map(t => met(e.out[t].me.R2.toFixed(3), `test R² ${c.short[t]}`, r2c(e.out[t].me.R2))).join("") + met(o.mt.R2.toFixed(3), "train R² oil", "train") + met(fmt(o.me.MAE, 1), "test MAE oil (STB/d)") + met(o.me.MAPE.toFixed(1) + " %", "test MAPE oil")
    : met(o.mt.R2.toFixed(3), "R² train", "train") + met(o.me.R2.toFixed(3), "R² test", "test") + met(fmt(o.me.MAE), `MAE test (${c.units[t0]})`) + met(fmt(o.me.RMSE), "RMSE test") + met(o.me.MAPE.toFixed(1) + " %", "MAPE test");
  const items = [], r = o.me.R2, gap = o.mt.R2 - r, cvb = m.cv.find(q => q.alg === m.alg);
  items.push(r < 0 ? ["b", `Test R² ${r.toFixed(2)} is below zero: the model is worse than always predicting the average.`] : r >= .95 ? ["g", `Test R² ${r.toFixed(3)}: excellent on unseen rows.`] : r >= .8 ? `Test R² ${r.toFixed(3)}: good, with room to improve.` : ["w", `Test R² ${r.toFixed(3)}: weak. Look at which rows the model misses in the cross-plot.`]);
  if (gap > .1) items.push(["w", `Training R² is ${gap.toFixed(2)} higher than test: the model memorises training rows (over-fitting). Try fewer trees levels, more neighbours, or a simpler model.`]);
  else if (o.mt.R2 < .8) items.push(["w", "Training R² is low too: the model is too simple for this data (under-fitting)."]);
  if (cvb && Math.abs(cvb.mean - r) > .08) items.push(["w", `Cross-validation promised ${cvb.mean.toFixed(3)}; the test gives ${r.toFixed(3)}. ${r < cvb.mean ? "The test set is harder than the folds (for example a held-out value at the edge)" : "The test set happens to be easier than the folds"}.`]);
  if (k === "vfm" && s.split.leak) items.push(["b", `${s.split.leak} test rows have copies in training, so these scores are optimistic (step 3).`]);
  if (k === "vfm") { const worst = c.targets.reduce((a, b) => e.out[b].me.R2 < e.out[a].me.R2 ? b : a); items.push(`Hardest phase: <b>${c.short[worst]}</b>. Each phase has its own model, so the phases can be compared directly.`); }
  if (e.imp.length) items.push(`The model leans most on <b>${e.imp[0][0]}</b>${e.imp.length > 1 ? `, then ${e.imp[1][0]}` : ""}. ${e.imp.filter(q => q[1] <= .002).length ? `${e.imp.filter(q => q[1] <= .002).map(q => q[0]).join(", ")} barely matter${k === "vfm" && e.imp.some(q => q[0] === "psep_psi" && q[1] <= .002) ? " (separator pressure does not change the rates)" : ""}.` : ""}`);
  $(`${k}-mean5`).innerHTML = meaning("What this means", items);
  const tr = e.tr, te = e.te, traces = [];
  if (c.targets.length > 1) c.targets.forEach((t, i) => { const mx = Math.max(...te.map(q => q[t])); traces.push({ x: te.map(q => q[t] / mx), y: e.out[t].pte.map(v => v / mx), mode: "markers", type: "scatter", name: `${c.short[t]}  R² ${e.out[t].me.R2.toFixed(3)}`, marker: { size: 5, color: [TE(), TR(), css("--good")][i], opacity: .75 } }); });
  else traces.push({ x: tr.map(q => q[t0]), y: o.ptr, mode: "markers", type: "scatter", name: `train R² ${o.mt.R2.toFixed(3)}`, marker: { size: 5, color: TR(), opacity: .5 } }, { x: te.map(q => q[t0]), y: o.pte, mode: "markers", type: "scatter", name: `test R² ${o.me.R2.toFixed(3)}`, marker: { size: 8, color: TE() } });
  const all = traces.flatMap(t => [...t.x, ...t.y]), lo = Math.min(...all), hi = Math.max(...all); traces.push({ x: [lo, hi], y: [lo, hi], mode: "lines", type: "scatter", name: "perfect", line: { color: MUTE(), dash: "dash" } });
  $(`${k}-ev-cross-read`).textContent = c.targets.length > 1 ? "Test wells only; each phase scaled to its largest value. Points on the dashed line are perfect." : "Points on the dashed line are perfect predictions. Petrol = training rows, amber = test rows.";
  Plotly.react(`${k}-ev-cross`, traces, layout({ xaxis: { title: { text: c.targets.length > 1 ? "measured (fraction of max)" : `measured ${t0}` } }, yaxis: { title: { text: "predicted" } } }), CFG);
  const ip = [...e.imp].reverse(); Plotly.react(`${k}-ev-imp`, [{ y: ip.map(q => q[0]), x: ip.map(q => q[1]), type: "bar", orientation: "h", marker: { color: TR() } }], layout({ margin: { l: 150, r: 14, t: 8, b: 44 }, xaxis: { title: { text: `drop in test R² (${c.short[t0]})` } } }), CFG);
  // physics check
  const med = Object.fromEntries(c.base.map(f => [f, median(s.clean.map(q => q[f]))])), ph = [];
  if (k === "asph") { const xs = Array.from({ length: 60 }, (_, i) => 3 + 22 * i / 59); [293, 308, 323, 332].forEach((T, i) => { ph.push({ x: xs, y: m.models[t0].predict(xs.map(x => m.F({ ...med, temperature_K: T, alkane_carbon_number: 7, dilution_ratio: x }))), mode: "lines", type: "scatter", name: `${T} K`, line: { color: CCOL[i], width: 2.5 } }); });
    const d7 = s.clean.filter(q => q.alkane_carbon_number === 7); ph.push({ x: d7.map(q => q.dilution_ratio), y: d7.map(q => q[t0]), mode: "markers", type: "scatter", name: "measured n-C7", marker: { size: 6, color: d7.map(q => CCOL[[293, 308, 323, 332].indexOf(q.temperature_K)]), line: { color: css("--surface"), width: 1 } } });
    $(`${k}-ev-phys-read`).textContent = "Model curves for n-heptane at four temperatures. Physics says: rising with dilution, levelling off, lower at higher temperature. Steps or wiggles mean the model is not physical between the data."; }
  else { const te = e.te, ch = [...new Set(te.map(q => q.choke_in))].sort((a, b) => a - b), cols = [TE(), TR(), MUTE()];
    c.targets.forEach((t, ti) => { const pte = e.out[t].pte; ph.push({ x: ch.map(String), y: ch.map(cv => { const idx = te.map((q, i) => q.choke_in === cv ? i : -1).filter(i => i >= 0); return 100 * mean(idx.map(i => Math.abs(pte[i] - te[i][t]))) / Math.max(1e-9, mean(idx.map(i => te[i][t]))); }), type: "bar", name: c.short[t], marker: { color: cols[ti] } }); });
    $(`${k}-ev-phys-read`).textContent = "Average error of the meter on the test wells for each choke size, as a share of the average rate at that choke. A tall bar at one choke means that part of the range is poorly learned."; }
  if (k === "vfm") Plotly.react(`${k}-ev-phys`, ph, layout({ barmode: "group", xaxis: { title: { text: "choke (in)" }, type: "category" }, yaxis: { title: { text: "mean absolute error (% of mean rate)" }, rangemode: "tozero" } }), CFG);
  else Plotly.react(`${k}-ev-phys`, ph, layout({ xaxis: { title: { text: "dilution ratio (mL/g)" } }, yaxis: { title: { text: "predicted wt%" } } }), CFG);
  regShapDraw(k); }
function regShapDraw(k) { const s = S[k], e = s.ev, c = REGCFG[k], m = s.model; if (!e || !e.shap) return; const sel = $(`${k}-shap-t`); if (sel) { sel.value = e.shapT; }
  const q = c.targets.indexOf(e.shapT), phi = shapPick(e.shap, q), imp = drawBeeswarm(`${k}-shap-bee`, phi, e.shapX, m.inputs), unit = c.units[e.shapT]; drawShapBar(`${k}-shap-bar`, imp, m.inputs, unit);
  const [top, col] = drawShapDep(`${k}-shap-dep`, phi, e.shapX, m.inputs, imp); $(`${k}-shap-dep-read`).textContent = `How the SHAP value of ${top} changes with its value${col ? `; colour is ${col}, the next most important input` : ""}. A curve here is the shape the model learned.`;
  const ord = imp.map((v, j) => [m.inputs[j], v]).sort((a, b) => b[1] - a[1]);
  $(`${k}-shap-note`).innerHTML = `Average prediction ${fmt(e.shap.base[q])} ${unit}. For ${e.shapX.length} test rows, each dot shows how much one input pushed that row's prediction up (right) or down (left). Largest average push: <b>${ord[0][0]}</b> (${fmt(ord[0][1])} ${unit})${ord[1] ? `, then ${ord[1][0]}` : ""}. Red dots on the right mean "high values raise the prediction".${ord[0][1] > .5 * Math.abs(e.shap.base[q]) ? ` <span class="warnline">Some pushes are larger than the average prediction itself. The inputs move together (see the correlation matrix in step 2), so SHAP mixes values from different rows that never occur together, and ${ALGN[m.alg]} has to extrapolate; the large pushes cancel out. Read the ranking, not the size, and compare with the permutation importance or a tree model.</span>` : ""}`;
  dcRefresh(`dc-${k}-5`); }

// ---------- step 6 · use ----------
function regUseControls(k) { if (k === "asph") return rangeCtl("au-T", "Temperature (K)", 280, 345, 1, 315, "Course data: 293 to 332 K") + `<div class="f"><label for="au-C">n-alkane</label><select id="au-C">${[5, 6, 7, 8, 9, 10, 11, 12].map(c => `<option value="${c}" ${c === 7 ? "selected" : ""}>n-C${c}${c === 11 ? " (not in the data)" : ""}</option>`).join("")}</select></div>`
    + rangeCtl("au-dr1", "Highest dilution ratio (mL/g)", 10, 40, 1, 25, "Course data reaches about 24");
  return `<div class="f"><label for="v-choke">Choke (in)</label><select id="v-choke">${[.25, .32, .42, .54, .7, .9, 1.16, 1.5, 1.94, 2.5].map(c => `<option ${c === .7 ? "selected" : ""}>${c}</option>`).join("")}</select></div>`
    + rangeCtl("v-bhp", "Bottom-hole pressure (psi)", 3900, 5500, 10, 4700) + rangeCtl("v-whp", "Wellhead pressure (psi)", 900, 4000, 10, 2000) + rangeCtl("v-wht", "Wellhead temperature (°F)", 80, 185, 1, 150)
    + rangeCtl("v-psep", "Separator pressure (psi)", 100, 600, 100, 300) + rangeCtl("v-tsep", "Separator temperature (°F)", 60, 142, 1, 100) + `<button class="btn ghost wide" id="v-snap" type="button">Fill in a real test well</button>`; }
function regUseInit(k) { if (k === "asph") { ["au-T", "au-dr1"].forEach(i => bind(i)); ["au-T", "au-dr1", "au-C"].forEach(i => $(i).addEventListener(i === "au-C" ? "change" : "input", () => W[k].ran[6] && liveUse(k))); return; }
  ["v-bhp", "v-whp", "v-wht", "v-psep", "v-tsep"].forEach(i => { bind(i); $(i).addEventListener("input", () => W[k].ran[6] && liveUse(k)); }); $("v-choke").addEventListener("change", () => W[k].ran[6] && liveUse(k));
  $("v-snap").addEventListener("click", () => { const s = S.vfm; if (!s || !s.ev) return; const r = s.ev.te[Math.floor(Math.random() * s.ev.te.length)]; $("v-choke").value = r.choke_in;
    [["v-bhp", "bhp_psi"], ["v-whp", "whp_psi"], ["v-wht", "wht_F"], ["v-psep", "psep_psi"], ["v-tsep", "tsep_F"]].forEach(([id, f]) => { $(id).value = r[f]; $("o-" + id).textContent = r[f]; }); s.snap = r; if (W.vfm.ran[6]) liveUse("vfm"); }); }
async function liveUse(k) { regUseRun(k); PROJ[k].stages[6].render(k); $(`${k}-stale6`).hidden = true; $(`${k}-res6`).classList.remove("dim"); dcRefresh(`dc-${k}-6`); }
function regUseRun(k) { const s = S[k], m = s.model;
  if (k === "asph") { const T = +$("au-T").value, C = +$("au-C").value, dr1 = +$("au-dr1").value, xs = Array.from({ length: 60 }, (_, i) => 2 + (dr1 - 2) * i / 59);
    s.use = { T, C, dr0: 2, dr1, xs, y: m.models.asphaltene_wt_pct.predict(xs.map(x => m.F({ temperature_K: T, alkane_carbon_number: C, dilution_ratio: x }))) }; return; }
  const g = { choke_in: +$("v-choke").value, bhp_psi: +$("v-bhp").value, whp_psi: +$("v-whp").value, wht_F: +$("v-wht").value, psep_psi: +$("v-psep").value, tsep_F: +$("v-tsep").value };
  const p = Object.fromEntries(REGCFG.vfm.targets.map(t => [t, m.models[t].predict([m.F(g)])[0]])), sc = VFM_F.map(f => { const v = s.clean.map(r => r[f]); return (Math.max(...v) - Math.min(...v)) || 1; });
  let best = null, bd = 1e9; s.clean.forEach(r => { const d = VFM_F.reduce((a, f, j) => a + ((r[f] - g[f]) / sc[j]) ** 2, 0); if (d < bd) { bd = d; best = r; } });
  const ch = [...new Set(s.clean.map(r => r.choke_in))].sort((a, b) => a - b), what = ["oil_rate_stb_d", "water_rate_bbl_d"].map(t => m.models[t].predict(ch.map(c => m.F({ ...g, choke_in: c }))));
  s.use = { g, p, best, dist: Math.sqrt(bd), ch, what }; }
function regUseRender(k) { const s = S[k], u = s.use, m = s.model;
  if (k === "asph") { const T = s.clean.map(r => r.temperature_K), inT = u.T >= Math.min(...T) && u.T <= Math.max(...T), inC = s.clean.some(r => r.alkane_carbon_number === u.C), plateau = u.y[u.y.length - 1];
    const at = dr => u.y[Math.round((dr - u.dr0) / (u.dr1 - u.dr0) * 59)];
    $("asph-met6").innerHTML = met(at(Math.min(10, u.dr1)).toFixed(2), "wt% at dilution 10", "test") + met(plateau.toFixed(2), `wt% at dilution ${u.dr1}`, "test") + met(inT ? "inside" : "outside", "temperature vs data", inT ? "good" : "warn") + met(inC ? "seen" : "new", "n-alkane", inC ? "good" : "warn");
    const items = [`For crude diluted with n-C${u.C} at ${u.T} K, the model predicts about <b>${at(Math.min(10, u.dr1)).toFixed(2)} wt%</b> at a dilution ratio of 10 and <b>${plateau.toFixed(2)} wt%</b> at ${u.dr1}.`];
    if (!inT) items.push(["w", `${u.T} K is outside the tested range (${Math.min(...T)} to ${Math.max(...T)} K). This is extrapolation: tree models flatten, polynomial models can run away.`]);
    if (!inC) items.push(["w", `n-C${u.C} was never tested${u.C === 11 ? ", but n-C10 and n-C12 were: the model must interpolate between them" : ""}.`]);
    if (u.dr1 > 25) items.push(["w", "Beyond a dilution ratio of about 24 there is no data."]);
    items.push(`Change the sliders: the curve follows. Compare with ${ALGN[m.alg]} vs another final model in step 4.`);
    $("asph-mean6").innerHTML = meaning("Prediction", items);
    const same = s.clean.filter(r => r.temperature_K === u.T && r.alkane_carbon_number === u.C), near = same.length ? same : s.clean.filter(r => r.alkane_carbon_number === u.C && Math.abs(r.temperature_K - u.T) === Math.min(...s.clean.filter(q => q.alkane_carbon_number === u.C).map(q => Math.abs(q.temperature_K - u.T))));
    Plotly.react("asph-use", [{ x: u.xs, y: u.y, mode: "lines", type: "scatter", name: `model · ${u.T} K, n-C${u.C}`, line: { color: TE(), width: 3 } }, { x: near.map(r => r.dilution_ratio), y: near.map(r => r.asphaltene_wt_pct), mode: "markers", type: "scatter", name: near.length ? `measured · ${near[0].temperature_K} K, n-C${u.C}` : "no measurements", marker: { size: 8, color: TR() } }],
      layout({ xaxis: { title: { text: "dilution ratio (mL/g)" } }, yaxis: { title: { text: "asphaltene precipitated (wt%)" }, rangemode: "tozero" }, shapes: [{ type: "rect", xref: "x", yref: "paper", x0: 24.5, x1: Math.max(24.6, u.dr1), y0: 0, y1: 1, fillcolor: css("--amber-soft"), opacity: .5, line: { width: 0 }, layer: "below" }] }), CFG);
    $("asph-usetab").innerHTML = tableHTML([4, 6, 8, 10, 15, 20, 25, 30, 35, 40].filter(d => d <= u.dr1).map(d => ({ "dilution ratio": d, "predicted wt%": +at(d).toFixed(3) })), 12); return; }
  const wc = u.p.water_rate_bbl_d / Math.max(1, u.p.oil_rate_stb_d + u.p.water_rate_bbl_d), gor = 1e6 * u.p.gas_rate_mmscf_d / Math.max(1, u.p.oil_rate_stb_d);
  $("vfm-met6").innerHTML = met(Math.max(0, u.p.oil_rate_stb_d).toFixed(0), "oil (STB/d)", "test") + met(Math.max(0, u.p.water_rate_bbl_d).toFixed(0), "water (bbl/d)", "train") + met(Math.max(0, u.p.gas_rate_mmscf_d).toFixed(2), "gas (MMscf/d)") + met((100 * wc).toFixed(0) + " %", "implied water cut") + met(gor.toFixed(0), "implied GOR (scf/STB)");
  const b = u.best, exact = u.dist < 1e-6, items = [`Gauges in, rates out: <b>${Math.max(0, u.p.oil_rate_stb_d).toFixed(0)} STB/d</b> oil, ${Math.max(0, u.p.water_rate_bbl_d).toFixed(0)} bbl/d water, ${Math.max(0, u.p.gas_rate_mmscf_d).toFixed(2)} MMscf/d gas.`,
    exact ? ["g", `This is a real test well: the separator measured ${b.oil_rate_stb_d.toFixed(0)} STB/d oil and ${b.water_rate_bbl_d.toFixed(0)} bbl/d water.`] : `Nearest real test (distance ${u.dist.toFixed(2)} in scaled gauge units) measured ${b.oil_rate_stb_d.toFixed(0)} STB/d oil and ${b.water_rate_bbl_d.toFixed(0)} bbl/d water.${u.dist > .25 ? " That is far: these gauge values are unusual, so trust the meter less." : ""}`];
  items.push(`The chart below runs the same meter on every test well and lays it over what the test separator measured.`);
  $("vfm-mean6").innerHTML = meaning("Meter reading", items);
  const TE_ = s.ev.te, step = Math.max(1, Math.floor(TE_.length / 120)), tr_ = [];
  [["oil_rate_stb_d", "oil", TE()], ["water_rate_bbl_d", "water", TR()]].forEach(([t, nm, col]) => { const tw = [...TE_].sort((a, b) => a[t] - b[t]).filter((_, i) => i % step === 0), pr = m.models[t].predict(tw.map(m.F)), xs = tw.map((_, i) => i + 1);
    tr_.push({ x: xs, y: tw.map(r => r[t]), name: `${nm}, separator`, mode: "markers", type: "scatter", marker: { size: 6, color: col, opacity: .5 } }, { x: xs, y: pr.map(v => Math.max(0, v)), name: `${nm}, meter`, mode: "lines", type: "scatter", line: { color: col, width: 2 } }); });
  Plotly.react("vfm-use", tr_, layout({ xaxis: { title: { text: "test wells, sorted by measured rate of each phase" } }, yaxis: { title: { text: "rate (STB/d or bbl/d)" }, rangemode: "tozero" } }), CFG); }

// ---------- data tables for the Data & code panels ----------
function regDC(k, i, N) { const s = S[k], c = REGCFG[k]; let tables = {}, note = "";
  if (i === 0) { tables = { [s.gen ? `generated_${k}.csv` : k === "asph" ? "asphaltene.csv" : "vfm_well_rates.csv"]: s.raw.map(r => { const o = { ...r }; delete o.suspect; if (k === "asph") delete o.series; if (k === "vfm") { delete o.case_id; delete o.test_id; delete o.water_cut; } return o; }) }; note = s.gen ? "generated in the browser" : "course data"; }
  const strip = R => k === "vfm" ? R.map(r => { const o = { ...r }; delete o.case_id; delete o.test_id; delete o.water_cut; return o; }) : R;
  if (i === 1) { tables = { "clean.csv": strip(s.clean), "removed_rows.csv": strip(s.removed) }; note = "after your cleaning decisions"; }
  if (i === 2) tables = { "split.csv": s.clean.map((r, j) => ({ row: j, split: s.split.test[j] ? "test" : "train", ...Object.fromEntries([...c.base, ...c.targets].map(f => [f, r[f]])) })) };
  if (i === 3) { const m = s.model, r = m.cv.find(q => q.alg === m.sel); tables = { "cross_validation.csv": m.cv.map(r => ({ model: ALGN[r.alg], mean_R2: r4(r.mean), std: r4(r.sd), MAE: r4(r.M.MAE), RMSE: r4(r.M.RMSE), ...Object.fromEntries(r.folds.map((v, j) => [`fold_${j + 1}`, r4(v)])) })), "fold_of_each_training_row.csv": m.y0.map((y, j) => ({ row: j, fold: m.fold[j] + 1, measured: y })) };
    if (r) tables[`out_of_fold_${r.alg}.csv`] = m.y0.map((y, j) => ({ row: j, fold: m.fold[j] + 1, measured: y, predicted: r4(r.oof[j]) })); }
  if (i === 4) tables = { "test_predictions.csv": s.ev.te.map((r, j) => ({ ...Object.fromEntries(s.model.inputs.map(f => [f, r4(c.extra[f] ? c.extra[f](r) : r[f])])), ...Object.fromEntries(c.targets.flatMap(t => [[`measured_${c.short[t]}`, r[t]], [`predicted_${c.short[t]}`, r4(s.ev.out[t].pte[j])]])) })), "importance.csv": s.ev.imp.map(([f, v]) => ({ input: f, drop_in_R2: r4(v) })) };
  if (i === 4 && s.ev.shap) { const q = c.targets.indexOf(s.ev.shapT), ph = shapPick(s.ev.shap, q); tables[`shap_values_${c.short[s.ev.shapT]}.csv`] = ph.map((r, j) => ({ ...Object.fromEntries(s.model.inputs.map((f, a) => [`${f}`, r4(s.ev.shapX[j][a])])), ...Object.fromEntries(s.model.inputs.map((f, a) => [`shap_${f}`, r4(r[a])])), base: r4(s.ev.shap.base[q]), prediction: r4(s.ev.shap.fx[j][q]) })); }
  if (i === 5) tables = k === "asph" ? { "predicted_curve.csv": s.use.xs.map((x, j) => ({ temperature_K: s.use.T, alkane_carbon_number: s.use.C, dilution_ratio: r4(x), predicted_wt_pct: r4(s.use.y[j]) })) }
    : { "meter_reading.csv": [{ ...s.use.g, ...Object.fromEntries(Object.entries(s.use.p).map(([t, v]) => [`virtual_${c.short[t]}`, r4(v)])) }], "what_if_choke.csv": s.use.ch.map((x, j) => ({ choke_in: x, virtual_oil: r4(s.use.what[0][j]), virtual_water: r4(s.use.what[1][j]) })) };
  return { tables, dataNote: note, code: pyReg(k, i), codeNote: pyNote(N), codeFile: `${k}_workflow_step${N + 1}.py` }; }

function regProject(k) { const asph = k === "asph";
  return { key: k, title: asph ? "Project 1 · Predict asphaltene precipitation" : "Project 3 · Build a virtual flow meter",
    intro: asph ? "How much asphaltene drops out when crude oil is diluted with an n-alkane? Seven steps from raw lab tests to a model you can ask about a new temperature or precipitant." : "Estimate oil, water and gas rates from the gauges a well already has. Seven steps from well tests to a meter you can read.",
    caseHTML: asph ? ASPH_CASE : VFM_CASE,
    stages: [
      { title: "Get the data", what: "Use the course tests or generate new ones with known physics.", controls: regDataControls(k), init: regDataInit, runLabel: "Load the data", empty: "Choose a data source and press <b>Load the data</b>.", run: regDataRun, render: regDataRender, dc: k2 => regDC(k2, 0, 0),
        charts: asph ? [{ id: "asph-curves", title: "Every test, one line per series", read: "A series is one temperature with one n-alkane, measured at several dilution ratios." }, { id: "asph-cn", title: "Heavier n-alkanes precipitate less", read: "Precipitate at the highest dilution of each series" }]
          : [{ id: "vfm-liq", title: "Liquid rate against choke size", read: "Colour is water cut" }, { id: "vfm-whp", title: "Oil rate against wellhead pressure", read: "Colour is choke size" }],
        task: asph ? ["Which curves look unlike the others?", "Why does precipitation level off at high dilution?"] : ["How many choke sizes were tested?", "Why do large chokes not give much more liquid?"] },
      edaStage(k),
      { title: "Clean the data", what: "Check for outliers and duplicates, then decide what to remove or add. Nothing is removed unless you tick it.", controls: regCleanControls(k), init: regCleanInit, onEnter: regCleanOnEnter, runLabel: "Check and apply", empty: "Tick the decisions you want (or none) and press <b>Check and apply</b>. The check shows what it found either way.", run: regCleanRun, render: regCleanRender, dc: k2 => regDC(k2, 1, 2),
        charts: [{ id: `${k}-clean-sc`, title: asph ? "Kept, removed and flagged tests" : "Does separator pressure change the rate?", read: asph ? "Rings = IQR flags, crosses = removed by your decisions" : "Eight well tests, each measured at six separator pressures. Flat lines: no effect." }, { id: `${k}-clean-hist`, title: "Distribution of the target", read: "" }, { id: `${k}-iqrtab-c`, title: "Ranges and IQR flags per column", read: "Read the ranges like an engineer: are they physical?", html: `<div class="scroll" id="${k}-iqrtab"></div>`, span: true }],
        task: asph ? ["Run once with no decisions, then remove the suspect tests. Which rows does the IQR rule miss?", "Would you delete data you cannot explain?"] : ["Why would copies of a test make a model look better than it is?"] },
      { title: "Split into train and test", what: "Hide some rows from the model. How you choose them decides what the test score means.", controls: regSplitControls(k), init: regSplitInit, onEnter: regHoldVals, runLabel: "Make the split", empty: "Choose how to pick test rows and press <b>Make the split</b>.", run: regSplitRun, render: regSplitRender, dc: k2 => regDC(k2, 2, 3),
        charts: [{ id: `${k}-split-sc`, title: "Which rows are test rows", read: "Petrol = training, amber = test" }, { id: `${k}-split-h`, title: "Do train and test cover the same range?", read: "Distribution of the target in each set" }],
        task: ["Which split matches how the model will be used in the field?"] },
      { title: "Compare and train", what: "Rank algorithms by cross-validation on the training rows only, then train the final model.", controls: regTrainControls(k), init: regTrainInit, onEnter: regTrainOnEnter, runLabel: "Compare and train", empty: "Choose inputs, algorithms to compare and a final model, then press <b>Compare and train</b>.", run: regTrainRun, render: regTrainRender, dc: k2 => regDC(k2, 3, 4),
        charts: [{ id: `${k}-kfold`, title: "How k-fold cross-validation uses the training rows", read: "One row per fold: amber rows are scored, petrol rows train the model. Each training row is scored exactly once.", cls: "short", span: true },
          { id: `${k}-cv`, title: "Cross-validation on training rows", read: "Bars are the mean R² over the folds; whiskers show the spread. Click a bar to see that algorithm.", cls: "tall" },
          { id: `${k}-cvfold`, title: "Score of every fold", read: "One line per algorithm; the selected one is amber. Lines that jump mean the score depends on which rows were held out.", cls: "tall" },
          { id: `${k}-alg-c`, title: "Results of one algorithm", read: "Click an algorithm to see its cross-validation results.", html: `<div class="row algbtns" id="${k}-algbtns"></div><div class="metrics" id="${k}-algmet"></div><p class="note" id="${k}-algnote"></p><button class="btn sm alt" id="${k}-algfinal" type="button">Make this the final model and train again</button>` },
          { id: `${k}-algpar`, title: "Out-of-fold predictions", read: "", cls: "tall" },
          { id: `${k}-curve`, title: "Neural network training loss", read: "Should fall and flatten; still falling means more epochs would help.", hidden: true }],
        task: ["Why must model choice not use the test rows?", "When two models tie, which would you deploy?"] },
      { title: "Evaluate on the test set", what: "Score the final model once on the hidden rows, then check it behaves like physics.", runLabel: "Evaluate the final model", empty: "Press <b>Evaluate the final model</b>. Do this once, after you have chosen the model.", run: regEvalRun, render: regEvalRender, dc: k2 => regDC(k2, 4, 5),
        init: k2 => { const el = $(`${k2}-shap-t`); if (el) el.addEventListener("change", () => { if (S[k2] && S[k2].ev) { S[k2].ev.shapT = el.value; regShapDraw(k2); } }); },
        charts: [{ id: `${k}-ev-cross`, title: "Predicted against measured", read: "", cls: "tall" }, { id: `${k}-ev-imp`, title: "Which inputs the model relies on (permutation)", read: "Drop in test R² when one input is shuffled", cls: "tall" }, { id: `${k}-ev-phys`, title: asph ? "Physics check" : "Error by choke size", read: "", span: true },
          { id: `${k}-shap-c`, title: "SHAP values: how each input pushes each prediction", read: "", span: true, html: `${asph ? "" : `<div class="row"><label for="${k}-shap-t">Explain</label><select id="${k}-shap-t">${REGCFG[k].targets.map(t => `<option value="${t}">${t}</option>`).join("")}</select></div>`}<p class="note" id="${k}-shap-note"></p><div class="plot tall" id="${k}-shap-bee"></div>` },
          { id: `${k}-shap-bar`, title: "Mean |SHAP| per input", read: "Average size of each input's push, in target units", cls: "tall" }, { id: `${k}-shap-dep`, title: "SHAP dependence of the top input", read: "", cls: "tall" }],
        task: ["Is the test score close to the cross-validation score? Why or why not?", "Would you trust the curve shape between the measured points?"] },
      { title: "Use the model", what: asph ? "Ask the model for a precipitation curve at conditions you choose." : "Enter gauge readings and read the virtual rates.", controls: regUseControls(k), init: regUseInit, runLabel: asph ? "Predict the curve" : "Read the meter", empty: asph ? "Choose a temperature and n-alkane, then press <b>Predict the curve</b>. After that the curve follows the sliders." : "Set the gauges and press <b>Read the meter</b>. After that the rates follow the sliders.", run: regUseRun, render: regUseRender, dc: k2 => regDC(k2, 5, 6),
        charts: asph ? [{ id: "asph-use", title: "Predicted precipitation curve", read: "Shaded: beyond the measured dilution ratios", cls: "tall" }, { id: "asph-usetab-c", title: "Values", read: "Read off at round dilution ratios", html: `<div class="scroll" id="asph-usetab"></div>` }]
          : [{ id: "vfm-use", title: "The meter against the test separator", read: "Test wells the model never saw, each phase sorted by its measured rate. Dots: separator. Lines: virtual meter.", cls: "tall", span: true }],
        task: asph ? ["Predict n-C11 at 315 K: no such test exists. Do you trust it?", "Go to 345 K. What does the model do outside the data?"] : ["Fill in a real test well and compare with the measured rates.", "Go back to step 4, untick whp_psi, and see what a failed gauge costs."] } ] }; }
