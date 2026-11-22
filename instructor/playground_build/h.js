"use strict";
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const CFG = { displayModeBar: false, responsive: true };
const sleep = () => new Promise(r => setTimeout(r, 0));
function layout(extra = {}) { const g = css("--plot-grid");
  const base = { margin: { l: 56, r: 14, t: 10, b: 46 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { family: "IBM Plex Sans, Segoe UI, Arial", size: 12.5, color: css("--ink") },
    xaxis: { gridcolor: g, zerolinecolor: g, linecolor: css("--line") }, yaxis: { gridcolor: g, zerolinecolor: g, linecolor: css("--line") }, legend: { orientation: "h", y: 1.14, x: 0, bgcolor: "rgba(0,0,0,0)" },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--line"), font: { color: css("--ink") } } };
  for (const k of Object.keys(extra)) base[k] = (typeof extra[k] === "object" && extra[k] !== null && !Array.isArray(extra[k]) && base[k]) ? Object.assign({}, base[k], extra[k]) : extra[k];
  return base; }
const TR = () => css("--petrol"), TE = () => css("--amber"), MUTE = () => css("--muted");
function bind(id, f = v => v) { const el = $(id), o = $("o-" + id); if (!el) return; const up = () => { if (o) o.textContent = f(+el.value); }; el.addEventListener("input", up); up(); return el; }
function seeded(seed) { let s = (seed * 2654435761) >>> 0 || 1; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function gauss(rng) { let u = 0; while (u === 0) u = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); }
function parseCSV(text) { const lines = text.trim().split(/\r?\n/).filter(l => l.trim()); const split = l => { const out = []; let cur = "", q = false; for (const ch of l) { if (ch === '"') q = !q; else if (ch === "," && !q) { out.push(cur); cur = ""; } else cur += ch; } out.push(cur); return out.map(s => s.trim()); };
  const head = split(lines[0]); return lines.slice(1).map(l => { const v = split(l), o = {}; head.forEach((h, i) => { const x = v[i]; o[h] = x !== undefined && x !== "" && !isNaN(+x) ? +x : (x ?? ""); }); return o; }); }
function tableHTML(rows, max = 12, cls = "") { if (!rows.length) return ""; const cols = Object.keys(rows[0]);
  return `<table class="${cls}"><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr>${rows.slice(0, max).map(r => `<tr>${cols.map(c => `<td class="${r._cls && r._cls[c] || ""}">${typeof r[c] === "number" ? +r[c].toFixed(4) : r[c]}</td>`).join("")}</tr>`).join("")}</table>`; }
const met = (v, l, c = "") => `<div class="m ${c}"><div class="v">${v}</div><div class="l">${l}</div></div>`;
const r2c = v => v >= 0.9 ? "good" : v >= 0.75 ? "warn" : "bad";
function meaning(title, items) { return `<h3>${title}</h3><ul>${items.filter(Boolean).map(i => Array.isArray(i) ? `<li class="${i[0]}">${i[1]}</li>` : `<li>${i}</li>`).join("")}</ul>`; }
// ---------- theme ----------
const RENDER = new Set();
function setTheme(t, store = true) { document.documentElement.dataset.theme = t; $("th-light").setAttribute("aria-pressed", t === "light"); $("th-dark").setAttribute("aria-pressed", t === "dark");
  if (store) try { localStorage.setItem("mlpg-theme", t); } catch (e) {}
  RENDER.forEach(f => { try { f(); } catch (e) { console.warn(e); } }); }
function initTheme() { let t = null; try { t = localStorage.getItem("mlpg-theme"); } catch (e) {}
  if (!t) t = window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  $("th-light").addEventListener("click", () => setTheme("light")); $("th-dark").addEventListener("click", () => setTheme("dark")); setTheme(t, false); }


// ===================== DATA & CODE PANELS (shared) =====================
const DC = {};
function dcEsc(s) { return String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
function dcCSV(rows) { if (!rows || !rows.length) return ""; const cols = Object.keys(rows[0]);
  return cols.join(",") + "\n" + rows.map(r => cols.map(c => { const v = r[c]; return typeof v === "string" && /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : (v ?? ""); }).join(",")).join("\n"); }
function dcFmt(v) { return typeof v === "number" ? (Number.isInteger(v) ? v : +v.toFixed(4)) : dcEsc(v ?? ""); }
function dcRegister(id, get) { DC[id] = { get, view: "data", table: null }; const el = document.getElementById(id);
  el.innerHTML = `<div class="dc-head"><h3>Data &amp; code behind this view</h3>
    <div class="dc-tabs" role="tablist"><button class="dc-b" data-v="data" aria-selected="true">Data</button><button class="dc-b" data-v="code" aria-selected="false">Python code</button></div>
    <select class="dc-sel" aria-label="table"></select><span class="dc-sp"></span>
    <button class="btn sm alt dc-copy">Copy</button><button class="btn sm alt dc-dl">Download</button></div>
    <div class="dc-note"></div><div class="dc-body"></div><div class="dc-toast"></div>`;
  el.querySelectorAll(".dc-b").forEach(b => b.addEventListener("click", () => { DC[id].view = b.dataset.v; dcRefresh(id); }));
  el.querySelector(".dc-sel").addEventListener("change", e => { DC[id].table = e.target.value; dcRefresh(id); });
  el.querySelector(".dc-copy").addEventListener("click", () => dcAct(id, "copy"));
  el.querySelector(".dc-dl").addEventListener("click", () => dcAct(id, "dl"));
  dcRefresh(id); }
function dcState(id) { const s = DC[id], g = s.get(), names = Object.keys(g.tables || {}); if (!s.table || !names.includes(s.table)) s.table = names[0]; return { s, g, names }; }
function dcRefresh(id) { if (!DC[id]) return; const el = document.getElementById(id), { s, g, names } = dcState(id);
  el.querySelectorAll(".dc-b").forEach(b => b.setAttribute("aria-selected", b.dataset.v === s.view));
  const sel = el.querySelector(".dc-sel"); sel.style.display = s.view === "data" && names.length > 1 ? "" : "none";
  sel.innerHTML = names.map(n => `<option ${n === s.table ? "selected" : ""}>${n}</option>`).join("");
  const body = el.querySelector(".dc-body"), note = el.querySelector(".dc-note");
  if (s.view === "code") { note.innerHTML = g.codeNote || ""; body.innerHTML = `<pre class="dc-code"><code></code></pre>`; body.querySelector("code").textContent = g.code || "# nothing yet"; return; }
  const rows = (g.tables || {})[s.table] || [];
  note.innerHTML = (g.dataNote ? g.dataNote + " · " : "") + (rows.length ? `${rows.length.toLocaleString()} rows × ${Object.keys(rows[0]).length} columns${rows.length > 200 ? " · showing the first 200; Copy and Download give every row" : ""}` : "No data yet");
  if (!rows.length) { body.innerHTML = ""; return; } const cols = Object.keys(rows[0]);
  body.innerHTML = `<div class="dc-table"><table><thead><tr>${cols.map(c => `<th>${dcEsc(c)}</th>`).join("")}</tr></thead><tbody>${rows.slice(0, 200).map(r => `<tr>${cols.map(c => `<td>${dcFmt(r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`; }
async function dcAct(id, act) { const el = document.getElementById(id), { s, g } = dcState(id), toast = el.querySelector(".dc-toast");
  const isCode = s.view === "code", text = isCode ? g.code : dcCSV((g.tables || {})[s.table] || []), name = isCode ? (g.codeFile || "lab_code.py") : s.table;
  if (!text) { toast.textContent = "Nothing to export yet."; return; }
  if (act === "copy") { try { await navigator.clipboard.writeText(text); toast.textContent = `${isCode ? "Code" : name} copied to the clipboard.`; } catch (e) { toast.textContent = "The browser blocked the clipboard. Use Download instead."; } return; }
  const dl = window.claude && window.claude.use ? await window.claude.use("downloads").catch(() => null) : null;
  if (dl) { try { await dl.save({ filename: name, data: text }); toast.textContent = `${name} saved.`; } catch (e) { toast.textContent = "The download was declined or is unavailable here. Use Copy instead."; } return; }
  try { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: isCode ? "text/x-python" : "text/csv" })); a.download = name; document.body.appendChild(a); a.click(); a.remove(); toast.textContent = `${name}: download started.`; }
  catch (e) { toast.textContent = "Downloads are blocked here. Use Copy instead."; } }
const py = v => typeof v === "boolean" ? (v ? "True" : "False") : typeof v === "string" ? JSON.stringify(v) : Array.isArray(v) ? "[" + v.map(py).join(", ") + "]" : String(+(+v).toFixed(4));

