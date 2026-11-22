// =====================================================================================
// WORKFLOW FRAMEWORK: seven steps per project, each unlocked by the one before
// =====================================================================================
const W = {}, PROJ = {};
function chartHTML(c) { return `<div class="card ${c.span ? "span" : ""}" id="${c.id}-card" ${c.hidden ? "hidden" : ""}><h3>${c.title}</h3>${c.read !== undefined ? `<p class="read" id="${c.id}-read">${c.read}</p>` : ""}${c.html || `<div class="plot ${c.cls || ""}" id="${c.id}"></div>`}</div>`; }
function buildProject(cfg) { const k = cfg.key; PROJ[k] = cfg; W[k] = { cur: 0, done: cfg.stages.map(() => false), ran: cfg.stages.map(() => false), stale: cfg.stages.map(() => false) };
  const sec = document.createElement("section"); sec.className = "lab"; sec.id = "lab-" + k;
  sec.innerHTML = `<div class="intro"><h2>${cfg.title}</h2><p>${cfg.intro}</p></div>
    ${cfg.caseHTML ? `<details class="casebox" open><summary>The case: problem, solution and data</summary><div class="casebody">${cfg.caseHTML}</div></details>` : ""}
    <nav class="stepper" id="${k}-stepper" aria-label="Workflow steps" style="--n:${cfg.stages.length}">${cfg.stages.map((s, i) => `<button data-i="${i}"><span class="dot">${i + 1}</span><span class="t">${s.title}</span></button>`).join("")}</nav>
    ${cfg.stages.map((s, i) => `<div class="stage" id="${k}-s${i}" ${i ? "hidden" : ""}><div class="grid">
      <aside class="rail"><div class="step"><p class="stage-title">Step ${i + 1} of ${cfg.stages.length}</p><h3>${s.title}</h3><p class="what">${s.what}</p>
        <div id="${k}-ctl${i}">${s.controls || ""}</div>
        <button class="btn wide" id="${k}-run${i}">${s.runLabel}</button><div class="prog"><i id="${k}-prog${i}"></i></div></div>
        ${s.task ? `<div class="task"><b>Think about it</b><ol>${s.task.map(t => `<li>${t}</li>`).join("")}</ol></div>` : ""}</aside>
      <div class="results">
        ${s.info ? `<div class="res-body" id="${k}-info${i}">${s.info}</div>` : ""}<div class="empty" id="${k}-empty${i}" ${s.info ? "hidden" : ""}><h3>${s.emptyTitle || "Not run yet"}</h3><p>${s.empty}</p><div class="ghost-charts"><div></div><div></div></div></div>
        <div class="stale" id="${k}-stale${i}" hidden><span id="${k}-stalemsg${i}">Settings changed since the last run; the results below are from the previous settings.</span><button class="btn sm" data-rerun="${k}-run${i}">${s.runLabel}</button></div>
        <div class="res-body" id="${k}-res${i}" hidden ${s.info ? 'data-info="1"' : ""}>
          <div class="metrics" id="${k}-met${i}"></div><div class="meaning" id="${k}-mean${i}"></div>
          ${s.charts && s.charts.length ? `<div class="charts">${s.charts.map(chartHTML).join("")}</div>` : ""}
          <div class="dc" id="dc-${k}-${i}"></div>
          ${i < cfg.stages.length - 1 ? `<div class="nextbar"><p class="note">${s.nextNote || ""}</p><button class="btn" data-goto="${i + 1}">Next: ${cfg.stages[i + 1].title}</button></div>` : ""}
        </div></div></div></div>`).join("")}`;
  $("main").appendChild(sec);
  sec.querySelectorAll(".stepper button").forEach(b => b.addEventListener("click", () => go(k, +b.dataset.i)));
  sec.querySelectorAll("[data-goto]").forEach(b => b.addEventListener("click", () => { go(k, +b.dataset.goto); window.scrollTo({ top: sec.offsetTop - 10, behavior: "smooth" }); }));
  sec.querySelectorAll("[data-rerun]").forEach(b => b.addEventListener("click", () => $(b.dataset.rerun).click()));
  cfg.stages.forEach((s, i) => {
    $(`${k}-run${i}`).addEventListener("click", () => runStage(k, i));
    const ctl = $(`${k}-ctl${i}`); const dirty = () => { if (W[k].ran[i]) { $(`${k}-stale${i}`).hidden = false; $(`${k}-res${i}`).classList.add("dim"); } };
    ctl.addEventListener("change", dirty); ctl.addEventListener("input", e => { if (e.target.type === "range") dirty(); });
    if (s.init) s.init(k);
    if (s.dc) dcRegister(`dc-${k}-${i}`, () => W[k].ran[i] ? s.dc(k) : { tables: {} });
  });
  stepper(k); }
function stepper(k) { const w = W[k]; document.querySelectorAll(`#${k}-stepper button`).forEach((b, i) => {
  const open = i === 0 || w.done[i - 1]; b.disabled = !open; b.classList.toggle("done", w.done[i]); b.classList.toggle("open", open && !w.done[i]); b.classList.toggle("cur", i === w.cur); b.classList.toggle("stale", w.stale[i] && !w.done[i]);
  b.querySelector(".dot").textContent = w.done[i] ? "✓" : (w.stale[i] ? "!" : i + 1); b.setAttribute("aria-current", i === w.cur ? "step" : "false");
  b.title = !open ? "Finish the step before first" : w.stale[i] ? "An earlier step changed: run this step again" : ""; }); }
function go(k, i) { const w = W[k]; if (i > 0 && !w.done[i - 1]) return; w.cur = i; PROJ[k].stages.forEach((_, j) => $(`${k}-s${j}`).hidden = j !== i); stepper(k);
  const s = PROJ[k].stages[i]; if (s.onEnter) s.onEnter(k); window.dispatchEvent(new Event("resize")); }
async function runStage(k, i) { const s = PROJ[k].stages[i], btn = $(`${k}-run${i}`); btn.disabled = true; $(`${k}-prog${i}`).style.width = "8%"; await sleep();
  try { await s.run(k); } catch (e) { console.error(e); $(`${k}-empty${i}`).hidden = false; $(`${k}-empty${i}`).querySelector("p").textContent = "This step could not run: " + e.message; btn.disabled = false; return; }
  const w = W[k]; w.ran[i] = true; w.done[i] = true; w.stale[i] = false;
  for (let j = i + 1; j < w.done.length; j++) if (w.ran[j] || w.done[j]) { w.done[j] = false; w.stale[j] = true; $(`${k}-stale${j}`).hidden = false; $(`${k}-stalemsg${j}`).textContent = `Step ${i + 1} changed, so these results are out of date. Run this step again.`; $(`${k}-res${j}`).classList.add("dim"); }
  $(`${k}-empty${i}`).hidden = true; $(`${k}-stale${i}`).hidden = true; $(`${k}-res${i}`).hidden = false; $(`${k}-res${i}`).classList.remove("dim");
  const draw = s.render; if (draw) { const f = () => { if (W[k].ran[i]) draw(k); }; RENDER.add(f); s._f = f; draw(k); }
  dcRefresh(`dc-${k}-${i}`); stepper(k); $(`${k}-prog${i}`).style.width = "100%"; setTimeout(() => $(`${k}-prog${i}`).style.width = "0", 350); btn.disabled = false; window.dispatchEvent(new Event("resize"));
  if (s.info) { $(`${k}-res${i}`).hidden = true; go(k, i + 1); const sec = $("lab-" + k); window.scrollTo({ top: sec.offsetTop - 10, behavior: "smooth" }); } }
async function prog(k, i, f) { $(`${k}-prog${i}`).style.width = (100 * f).toFixed(0) + "%"; await sleep(); }
// ---------- shared helpers for the projects ----------
function srcToggle(k) { return `<div class="src-toggle" role="group" aria-label="Data source"><button id="${k}-src-course" aria-pressed="true">Course data</button><button id="${k}-src-gen" aria-pressed="false">Generate new data</button></div>`; }
function initSrc(k) { const set = g => { $(`${k}-src-course`).setAttribute("aria-pressed", !g); $(`${k}-src-gen`).setAttribute("aria-pressed", g); $(`${k}-genbox`).hidden = !g; $(`${k}-coursebox`).hidden = g;
    if (W[k].ran[0]) { $(`${k}-stale0`).hidden = false; $(`${k}-res0`).classList.add("dim"); } };
  $(`${k}-src-course`).addEventListener("click", () => set(false)); $(`${k}-src-gen`).addEventListener("click", () => set(true)); }
const isGen = k => $(`${k}-src-gen`).getAttribute("aria-pressed") === "true";
function rangeCtl(id, label, mn, mx, st, val, hint) { return `<div class="f"><label for="${id}">${label}</label><output id="o-${id}"></output><input type="range" id="${id}" min="${mn}" max="${mx}" step="${st}" value="${val}">${hint ? `<span class="hint">${hint}</span>` : ""}</div>`; }
function decision(id, label, text, checked) { return `<div class="decision"><label class="check"><input type="checkbox" id="${id}" ${checked ? "checked" : ""}> ${label}</label>${text ? `<p>${text}</p>` : ""}</div>`; }
function kfold(n, K, rng, groups) { // returns array of fold index per row; groups keep rows together
  const fold = new Array(n);
  if (groups) { const g = [...new Set(groups)], ord = shuffleIdx(g.length, rng), map = new Map(); ord.forEach((gi, j) => map.set(g[gi], j % K)); groups.forEach((gv, i) => fold[i] = map.get(gv)); }
  else shuffleIdx(n, rng).forEach((r, j) => fold[r] = j % K);
  return fold; }
function stratFolds(y, K, rng) { const fold = new Array(y.length); const by = {}; y.forEach((c, i) => (by[c] = by[c] || []).push(i));
  let off = 0; Object.values(by).forEach(ix => { shuffleIdx(ix.length, rng).forEach((j, q) => fold[ix[j]] = (q + off) % K); off += ix.length; }); return fold; }
const sd = a => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) ** 2))); };
const msec = s => s < .001 ? "< 1 ms" : s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(1)} s`;
