<div align="center">

# Machine Learning for Petroleum Engineering

**Course code, field datasets and an interactive playground for applied machine learning in the oil and gas industry**

[![Live demo](https://img.shields.io/badge/live%20demo-GitHub%20Pages-0B6E79)](https://matinshahin.github.io/ml-petroleum-engineering/)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](#installation)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Notebooks](https://img.shields.io/badge/labs-6%20Jupyter%20notebooks-F37626?logo=jupyter&logoColor=white)](notebooks)

[**Open the interactive playground**](https://matinshahin.github.io/ml-petroleum-engineering/) · [Labs](#labs) · [Datasets](#datasets) · [Python package](#python-package-aiml) · [Author](#author)

</div>

---

## Overview

This repository accompanies **AI in Oil & Gas, Part I: AI and Machine Learning Fundamentals**, a course for petroleum and reservoir engineers taught by Matin Shahin (AI Innovators). It covers the full supervised-learning workflow on three petroleum engineering problems:

| Project | Task | Data | Question |
|---|---|---|---|
| **Asphaltene precipitation** | Regression | 167 laboratory tests | How much asphaltene drops out when crude is diluted with an n-alkane at a given temperature? |
| **EOR screening** | Classification (6 classes) | 455 field projects | Which enhanced oil recovery method suits a reservoir, compared with the Taber screening criteria? |
| **Virtual flow metering** | Multi-output regression | 2,100 well tests | Can oil, water and gas rates be read continuously from the gauges a well already has? |

Every project is available in three forms: the **data** (CSV), the **code** (a Python package and Jupyter labs with solutions) and a **browser tool** that runs the same workflow step by step without installing anything.

## Interactive playground

<https://matinshahin.github.io/ml-petroleum-engineering/>

The playground walks through a seven-step workflow for each project. Nothing runs until the user presses a button, and each step unlocks after the previous one.

1. **Get the data**: load the course dataset, or generate a new one from a physics-based model.
2. **Explore the data (EDA)**: distributions, Pearson or Spearman correlation matrix, standardised box plots, each input against the target, summary statistics.
3. **Clean**: IQR outliers, duplicated well tests, physically impossible values. The learner decides what to remove.
4. **Split**: random, by group (unseen well tests or curves), or hold out one value, with an automatic leakage check.
5. **Compare and train**: k-fold cross-validation (k = 3 to 10) of up to seven algorithms. The tool shows the folds, the score of every fold, and the out-of-fold results of any algorithm you click.
6. **Evaluate on the test set**: cross-plots or confusion matrix, permutation importance, and SHAP values (beeswarm, mean |SHAP|, dependence).
7. **Use the model**: predict a precipitation curve, screen a new field, or read the virtual flow meter.

Each step has a *Data & code* panel with the step's table as CSV and the equivalent Python script, so every result in the browser can be reproduced in a notebook. Light and dark themes are available. For offline use, open `playground/ml_playground_offline.html` with `plotly.min.js` in the same folder.

## Repository structure

```text
ml-petroleum-engineering/
├── aiml/                  Python package used by the labs and the generated code
│   ├── data.py            loaders for the three datasets
│   ├── prep.py            IQR and z-score outlier detection
│   ├── models.py          ready-to-use regressors and classifiers with correct scaling
│   ├── evaluate.py        metrics, model comparison, cross-plots, confusion matrix, importance
│   └── generators.py      physics-based data generators and the Taber screening rules
├── data/                  course datasets (CSV); data/original/ holds the source Excel files
├── notebooks/             six labs with exercises
├── solutions/             the same labs, solved and executed
├── playground/            offline version of the interactive tool
├── site/                  the page published on GitHub Pages
└── instructor/            build scripts for the notebooks and the playground
```

## Installation

```bash
git clone https://github.com/MatinShahin/ml-petroleum-engineering.git
cd ml-petroleum-engineering
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Python 3.9 or later is required. `shap` is only needed for the SHAP sections. The labs also run on Google Colab: upload the repository and install `requirements.txt` in the first cell.

## Quick start

```python
from sklearn.model_selection import GroupShuffleSplit
from aiml.data import load_vfm
from aiml.models import regressors
from aiml.evaluate import compare

df = load_vfm()                                    # adds case_id: copies of one well test share it
X = df[["choke_in", "bhp_psi", "whp_psi", "wht_F", "psep_psi", "tsep_F"]]
y = df["oil_rate_stb_d"]

# split by well test, so no copy of a test is on both sides
train, test = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(X, y, groups=df.case_id))
print(compare(regressors(), X.iloc[train], y.iloc[train], X.iloc[test], y.iloc[test]))
```

Generate a new dataset from the command line:

```bash
python -m aiml.generators vfm --n 1000 --noise 0.02 --out my_well_tests.csv
```

## Labs

| Lab | Topic | Main ideas |
|---|---|---|
| 0 | Python and data preparation | descriptive statistics, IQR outliers, duplicates, log transforms, scaling, stratified split |
| 1 | Regression: asphaltene precipitation | ten algorithms, data cleaning versus algorithm choice, cross-validated tuning, unseen-series test, physics features |
| 2 | Classification: EOR screening | imbalanced classes, micro, macro and weighted F1, confusion matrix, machine learning versus Taber rules |
| 3 | Virtual flow metering | data leakage from duplicated tests, group split, three phases, extrapolation to an unseen choke, failed-gauge scenario |
| 4 | Bias, variance and explainability | regularisation, learning curves, partial dependence, SHAP, k-means clustering |
| 5 | Generate your own data | gauge noise, number of tests, simulation-to-field transfer, planted outliers |

`notebooks/` contains the student versions with exercises; `solutions/` contains the executed answers.

## Datasets

| File | Rows | Inputs | Target |
|---|---|---|---|
| `asphaltene.csv` | 167 | temperature (K), n-alkane carbon number (C5 to C12), dilution ratio (mL/g) | asphaltene precipitated (wt%) |
| `eor_screening.csv` | 455 | permeability, depth, API gravity, viscosity, temperature, oil saturation | EOR method (CO2, combustion, hydrocarbon gas, N2, polymer, steam) |
| `vfm_well_rates.csv` | 2,100 | choke, BHP, WHP, WHT, separator pressure and temperature | oil, water and gas rates |

The datasets keep their real data-quality issues on purpose, because each one teaches something:

- **Asphaltene**: 17 tests at 332 K with n-C6 to n-C8 are about ten times higher than every other series. They are flagged, not removed.
- **EOR screening**: 45 projects report 0 cP viscosity, one reservoir is at 5 °F, and 43 rows are duplicates.
- **Virtual flow metering**: each well test is repeated at six separator pressures with identical rates, so the 2,100 rows hold 972 distinct tests. A random split leaks copies into the test set.

## Python package `aiml`

| Module | Contents |
|---|---|
| `aiml.data` | `load_asphaltene()`, `load_eor()`, `load_vfm()` |
| `aiml.prep` | `detect_outliers_iqr()`, `remove_outliers()`, `outlier_table()`, `zscore_outliers()`, `plot_outliers()` |
| `aiml.models` | `regressors()` and `classifiers()`: linear or logistic regression, polynomial, KNN, SVM, decision tree, random forest, extra trees, AdaBoost, gradient boosting, histogram gradient boosting, MLP and a voting ensemble |
| `aiml.evaluate` | `report()`, `compare()`, `crossplot()`, `hyper_curve()`, `confusion()`, `importance()` |
| `aiml.generators` | `make_asphaltene_data()`, `make_vfm_data()` (inflow, tubing and Gilbert choke model), `make_eor_data()`, `taber_screen()` |

## Related repository

[**data-driven-reservoir-modeling**](https://github.com/MatinShahin/data-driven-reservoir-modeling): Part II of the course, covering smart proxy modeling and top-down modeling of reservoirs.

## Citation

If you use this material in teaching or research, please cite it as:

```bibtex
@misc{shahin2026mlpetroleum,
  author       = {Shahin, Matin},
  title        = {Machine Learning for Petroleum Engineering: course code, datasets and interactive playground},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/MatinShahin/ml-petroleum-engineering}}
}
```

## Author

**Matin Shahin**
Founder, Chair & CTO, AI Innovators · Principal Applied AI Research Scientist

- GitHub: [@MatinShahin](https://github.com/MatinShahin)
- Email: matin.shahin97@gmail.com

Questions, corrections and suggestions are welcome through [GitHub Issues](https://github.com/MatinShahin/ml-petroleum-engineering/issues).
