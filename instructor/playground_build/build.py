"""Assemble the Machine Learning Playground (workflow edition) into one self-contained HTML page."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); KIT = os.path.dirname(HERE)
css = open(os.path.join(HERE, "style.css")).read() + open(os.path.join(HERE, "fonts.css")).read()
data = "".join(f"window.{k} = {json.dumps(open(os.path.join(KIT, 'data', f)).read())};\n" for k, f in
               [("CSV_ASPH", "asphaltene.csv"), ("CSV_EOR", "eor_screening.csv"), ("CSV_VFM", "vfm_well_rates.csv")])
js = "\n".join(open(os.path.join(HERE, f)).read() for f in ["h.js", "ml.js", "prefix.js", "wf.js", "cases.js", "eda.js", "reg.js", "eor.js"])
boot = """
initTheme();
buildProject(regProject("asph")); buildProject(eorProject()); buildProject(regProject("vfm"));
document.addEventListener("change", e => { if (e.target && e.target.id === "eor-cmn") drawEorCM(); });
RENDER.add(() => S.eor && S.eor.ev && W.eor.ran[5] && drawEorCM());
function show(lab) { document.querySelectorAll(".lab").forEach(s => s.classList.toggle("on", s.id === "lab-" + lab)); document.querySelectorAll("nav.tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.lab === lab)); window.dispatchEvent(new Event("resize")); }
document.querySelectorAll("nav.tabs button").forEach(b => b.addEventListener("click", () => { try { history.replaceState(null, "", "#" + b.dataset.lab); } catch (e) {} show(b.dataset.lab); }));
(function () { const h = (location.hash || "").slice(1); show(["asph", "eor", "vfm"].includes(h) ? h : "asph"); })();
"""
import base64
L64 = "data:image/png;base64," + base64.b64encode(open(os.path.join(HERE, "logo_light_bg.png"), "rb").read()).decode()
D64 = "data:image/png;base64," + base64.b64encode(open(os.path.join(HERE, "logo_dark_bg.png"), "rb").read()).decode()
body = open(os.path.join(HERE, "body2.html")).read().replace("{{LOGO_L}}", L64).replace("{{LOGO_D}}", D64)
page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Machine Learning for Petroleum Engineering · AI Innovators</title>
<script>try{{var t=localStorage.getItem("mlpg-theme")||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");document.documentElement.dataset.theme=t;}}catch(e){{}}</script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{css}</style>
</head>
<body>
{body}
<script>
{data}</script>
<script>
{js}
{boot}</script>
</body>
</html>
"""
open(os.path.join(HERE, "index.html"), "w").write(page)
cdn = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
open(os.path.join(HERE, "ml_playground_offline.html"), "w").write(page.replace(cdn, '<script src="plotly.min.js"></script>'))
print("index.html", len(page) // 1024, "KB")
