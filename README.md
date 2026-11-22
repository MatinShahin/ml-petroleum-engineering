# Machine Learning for Petroleum Engineering

Developed by **Matin Shahin** · AI Innovators

**Live tool:** https://matinshahin.github.io/ml-petroleum-engineering/ (GitHub Pages, deployed automatically from `site/` on every push to `main`)

# AI in Oil & Gas · Part I · AI & ML Fundamentals: course kit

Instructor: Matin Shahin · AI Innovators

Every topic comes in three formats: **data** (CSV files), **code** (Python package + Jupyter labs) and a **tool**
(the Machine Learning Playground, a web page that runs in any browser).

## 1 · Data (`data/`)

| File | Rows | Task | Inputs → target |
|---|---|---|---|
| `asphaltene.csv` | 167 | regression | `temperature_K`, `alkane_carbon_number` (n-C5 … n-C12 precipitant), `dilution_ratio` (mL/g) → `asphaltene_wt_pct` |
| `eor_screening.csv` | 455 | classification | `permeability_md`, `depth_ft`, `api_gravity`, `viscosity_cp`, `temperature_F`, `oil_saturation_pct` → `eor_method` (6 methods) |
| `vfm_well_rates.csv` | 2,100 | regression | `choke_in`, `bhp_psi`, `whp_psi`, `wht_F`, `psep_psi`, `tsep_F` → `oil_rate_stb_d`, `water_rate_bbl_d`, `gas_rate_mmscf_d` |

`data/original/` keeps the original Excel files and `outlier_detection.py` from the course.

Data-quality notes that the labs use as lessons:
* **Asphaltene**: 17 tests (332 K with n-C6, C7, C8) are about ten times higher than every other series. They are kept and flagged (`load_asphaltene()` adds `suspect`).
* **EOR**: 45 projects report 0 cP viscosity, one reservoir is at 5 °F, 43 rows are duplicates. Columns were renamed (`Temprature` → `temperature_F`, `Saturation` → `oil_saturation_pct`) and `combustion` → `Combustion`.
* **VFM**: every well test is repeated for six separator pressures with identical rates, so the 2,100 rows hold fewer than 1,000 different tests. A random split leaks copies into the test set; split by `case_id` (`load_vfm()`).

## 2 · Code

### Python package `aiml/`
| Module | What it gives you |
|---|---|
| `aiml.data` | `load_asphaltene()`, `load_eor()`, `load_vfm()`, feature and target names |
| `aiml.prep` | `detect_outliers_iqr`, `remove_outliers` (from the course's `outlier_detection.py`), `outlier_table`, `plot_outliers` |
| `aiml.models` | `regressors()` and `classifiers()`: the algorithms of the course notebooks (linear, polynomial, KNN, SVR/SVC, tree, random forest, extra trees, AdaBoost, gradient boosting, histogram boosting, MLP, voting), with correct scaling pipelines |
| `aiml.evaluate` | `compare`, `crossplot`, `hyper_curve`, `confusion`, `importance`, regression and classification metrics |
| `aiml.generators` | `make_asphaltene_data`, `make_vfm_data` (IPR + tubing + Gilbert choke), `make_eor_data`, `taber_screen` (Taber screening rules) |

Make new data from a terminal: `python -m aiml.generators vfm --n 1000 --noise 0.02 --out my_tests.csv`

### Labs (`notebooks/` for learners, `solutions/` already run)
| Lab | Topic | Slides |
|---|---|---|
| Lab 0 | Python and data preparation: describe, IQR outliers, duplicates, log transform, scaling, stratified split | Modules 04, 08 |
| Lab 1 | Regression on asphaltene: ten algorithms, cleaning vs algorithm choice, tuning by cross-validation, unseen-series test, physics feature | Modules 06, 07 |
| Lab 2 | Classification on EOR screening: imbalanced classes, micro/macro/weighted F1, confusion matrix, ML vs Taber rules, screen a new field | Modules 06, 07 |
| Lab 3 | Virtual flow metering: the leakage trap, group split, three phases, unseen choke, what-if, failed gauge | Modules 03, 06, 07 |
| Lab 4 | Bias, variance, regularization, partial dependence, k-means clustering | Modules 06, 07 |
| Lab 5 | Generate your own data: gauge noise, how many tests, sim-to-real, planted outliers | — |

Install: `pip install -r requirements.txt`, then open the notebooks with Jupyter (or upload the kit to Google Colab).

## 3 · Tool: the Machine Learning Playground
Open `playground/ml_playground_offline.html` in a browser (keep `plotly.min.js` next to it).

Three projects (asphaltene, EOR screening, virtual flow meter), each a seven-step workflow: **get the data** → **explore the data (EDA)** (distributions, correlation matrix (Pearson or Spearman), spread and outliers, each input against the target, summary statistics) → **clean** (outliers, duplicates, impossible values; the learner decides what to remove or add) → **split** (random, unseen groups or hold-out, with a leakage check) → **compare and train** (k-fold cross-validation with k from 3 to 10: a picture of the folds, the score of every fold, and a click on any algorithm shows its own results: metrics, fold scores and out-of-fold predictions or confusion matrix) → **evaluate on the test set** (cross-plot or confusion matrix, permutation importance, SHAP values: beeswarm, mean |SHAP| and dependence plot) → **use the model**. Each project opens with the case: the problem, the solution and the data, plus where the method is used in practice. The SHAP code in the Python panels needs `pip install shap`.

Nothing runs until you press a button; each step unlocks after the one before; changing an earlier step marks later steps out of date. Every step has a "What this means" note and a Data & code panel with the step's table (CSV) and the Python for the workflow so far. Light and dark themes in the top right.

`instructor/` holds the scripts that build the notebooks and the Playground.
