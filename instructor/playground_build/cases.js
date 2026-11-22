// ---------- the case behind each project ----------
const flow = steps => `<div class="flow">${steps.map((s, i) => `<span>${i + 1} · ${s}</span>${i < steps.length - 1 ? "<i>→</i>" : ""}`).join("")}</div>`;
const STEPS6 = (use) => flow(["Get the data", "Explore the data (EDA)", "Clean the data", "Split into train and test", "Compare and train", "Evaluate on the test set", use]);
const ASPH_CASE = `<div class="case">
  <div class="card problem"><h3>The problem</h3><p>Asphaltenes are the heaviest part of crude oil. When the oil is diluted with a light hydrocarbon (n-alkane), cooled or depressurised, they <b>precipitate</b> and deposit in the reservoir, the tubing and the facilities, plugging wells and pipelines.</p>
    <p>Before a solvent injection or a gas-lift design, engineers need to know how much will precipitate. Each laboratory test (titrating the oil with an n-alkane at one temperature and one dilution ratio) takes hours to days.</p></div>
  <div class="card solution"><h3>The solution: a regression model</h3><p>Use the tests already done to train a model that predicts the weight percent precipitated for any temperature, n-alkane and dilution ratio inside the tested range, then check that its curves behave like the physics: rising with dilution, levelling off, and falling for heavier n-alkanes and higher temperatures.</p></div>
  <div class="card data"><h3>The case and its data</h3><table class="kv"><tr><td>Tests</td><td>167 laboratory measurements on one crude oil</td></tr><tr><td>Temperature</td><td>293, 308, 323 and 332 K</td></tr><tr><td>Precipitant</td><td>n-pentane to n-dodecane (carbon number 5 to 12)</td></tr>
    <tr><td>Dilution ratio</td><td>about 2 to 24 mL of n-alkane per g of oil</td></tr><tr><td>Target</td><td>asphaltene precipitated, wt %</td></tr><tr><td>Known issue</td><td>17 tests (332 K, n-C6 to n-C8) about ten times higher than all others</td></tr></table></div>
</div>
${STEPS6("Use the model")}
<div class="card"><h3>Where this is used in practice</h3><ul class="refs">
  <li>Standard laboratory methods (for example IP 143 and ASTM D6560) measure asphaltene content by precipitation with n-heptane; titration with different n-alkanes and dilution ratios gives curves like the ones in this dataset.</li>
  <li>Scaling-equation models (Rassamdana and co-workers, AIChE Journal, 1996) use the same three inputs (dilution ratio, n-alkane carbon number, temperature); machine-learning models are now compared against them.</li>
  <li>Typical uses: screening solvents for asphaltene-prone oils, designing CO₂ or gas injection, and choosing inhibitor treatments.</li></ul></div>`;
const EOR_CASE = `<div class="case">
  <div class="card problem"><h3>The problem</h3><p>A reservoir is past its primary and waterflood life. Which <b>enhanced oil recovery</b> method should the team study first: gas injection (CO₂, hydrocarbon, nitrogen), chemical (polymer) or thermal (steam, in-situ combustion)? A full feasibility study for each method costs months.</p></div>
  <div class="card solution"><h3>The solution: a classification model</h3><p>Learn from projects already run worldwide which method was chosen for which reservoir, using six properties every field has. Compare the model with the classic expert screening rules, then screen a new field and read both answers.</p></div>
  <div class="card data"><h3>The case and its data</h3><table class="kv"><tr><td>Projects</td><td>455 field EOR projects</td></tr><tr><td>Inputs</td><td>permeability, depth, API gravity, oil viscosity, temperature, oil saturation</td></tr>
    <tr><td>Target</td><td>EOR method: CO₂, hydrocarbon gas, nitrogen, polymer, steam, combustion</td></tr><tr><td>Known issues</td><td>unbalanced classes (7 nitrogen vs 197 steam), zero viscosities, duplicates</td></tr><tr><td>Baseline</td><td>Taber, Martin and Seright screening criteria</td></tr></table></div>
</div>
${STEPS6("Screen a new field")}
<div class="card"><h3>Where this is used in practice</h3><ul class="refs">
  <li>The screening criteria of Taber, Martin and Seright (SPE Reservoir Engineering, 1997) are the industry's best-known rules for matching EOR methods to reservoir properties.</li>
  <li>Worldwide EOR projects are reported in the Oil &amp; Gas Journal's EOR surveys; databases of that kind are what screening models learn from.</li>
  <li>Data-driven screening is used to shortlist methods before laboratory and simulation studies, and to spot analogue fields.</li></ul></div>`;
const VFM_CASE = `<div class="case">
  <div class="card problem"><h3>The problem</h3><p>Oil, water and gas rates of each well are measured only when the well is routed through a <b>test separator</b>, often once a month, and subsea wells share flowlines. Between tests, the operator does not know how much each well produces, which matters for allocation, reservoir management and spotting problems early.</p></div>
  <div class="card solution"><h3>The solution: a virtual flow meter</h3><p>Learn the rates from the gauges every well already has (choke, bottom-hole and wellhead pressure and temperature, separator conditions), using the well tests as training data. Then read the rates continuously from the gauges.</p></div>
  <div class="card data"><h3>The case and its data</h3><table class="kv"><tr><td>Tests</td><td>2,100 rows of well tests at 10 choke sizes</td></tr><tr><td>Inputs</td><td>choke, BHP, WHP, WHT, separator pressure and temperature</td></tr>
    <tr><td>Targets</td><td>oil (STB/d), water (bbl/d), gas (MMscf/d)</td></tr><tr><td>Known issue</td><td>each test repeated for 6 separator pressures with the same rates: about 970 different tests</td></tr></table></div>
</div>
${STEPS6("Read the meter")}
<div class="card"><h3>Where this is used in practice</h3><ul class="refs">
  <li>Virtual flow metering is in routine use on offshore fields, for example in the North Sea, both as physics-based models (well and choke models) and as data-driven models trained on well tests.</li>
  <li>Data-driven meters are cheaper to build but only reliable inside the conditions they were trained on; that is why this project tests a choke size the model never saw.</li>
  <li>Typical uses: daily production allocation, back-allocation checks against fiscal meters, and early detection of water breakthrough or gas coning.</li></ul></div>`;
