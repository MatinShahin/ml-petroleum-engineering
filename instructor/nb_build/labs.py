from common import build, SETUP
import nbformat as nbf, os, sys

LABS = {}

# =====================================================================================
LABS["Lab0_Python_and_Data_Preparation"] = [
("md", """
# Lab 0 · Python and data preparation
**Course:** AI in Oil & Gas · Part I, AI & ML Fundamentals  ·  **Time:** 60 min  ·  **Slides:** Modules 04 and 08

Before any algorithm, look at the data. In this lab you load the three course datasets, find what is wrong with
them, and prepare them so a model can learn.

| Dataset | File | Task | Rows | Inputs → target |
|---|---|---|---|---|
| Asphaltene precipitation | `asphaltene.csv` | regression | 167 | temperature, n-alkane carbon number, dilution ratio → wt% precipitated |
| EOR screening | `eor_screening.csv` | classification | 455 | permeability, depth, API, viscosity, temperature, oil saturation → EOR method |
| Virtual flow metering | `vfm_well_rates.csv` | regression | 2,100 | choke, BHP, WHP, WHT, Psep, Tsep → oil, water, gas rates |

**You will:** load and describe data · plot distributions · detect outliers with the IQR rule · transform skewed inputs · scale · split.
"""),
("code", SETUP + """
from aiml.data import load_asphaltene, load_eor, load_vfm, ASPH_FEATURES, ASPH_TARGET, EOR_FEATURES
from aiml.prep import detect_outliers_iqr, outlier_table, plot_outliers
asph = pd.read_csv(f"{DATA}/asphaltene.csv")
eor = pd.read_csv(f"{DATA}/eor_screening.csv")
vfm = pd.read_csv(f"{DATA}/vfm_well_rates.csv")
asph.head()
"""),
("md", "## 1 · First look\n`info()` shows types and missing values; `describe()` the ranges. Read them like an engineer: are the units and ranges physical?"),
("code", """
asph.info(); asph.describe().T
"""),
("code", """
eor.describe().T.round(1)
"""),
("md", """
### ✏️ Your turn 1
In `eor`, find (a) how many projects report a viscosity of **0 cP**, (b) the lowest reservoir temperature, and
(c) how many rows are exact duplicates in the six inputs and the label. Which of these are data errors?
"""),
("ex", """
# (a) viscosity == 0
# (b) minimum temperature
# (c) duplicated rows (inputs + label)
""", """
print("viscosity = 0 cP:", (eor.viscosity_cp == 0).sum(), "projects (a rounded value; real oil is never 0 cP)")
print("lowest temperature:", eor.temperature_F.min(), "°F (a 5 °F reservoir is not physical: likely a typing error)")
print("duplicates:", eor.duplicated(subset=EOR_FEATURES + ["eor_method"]).sum(), "rows (same field reported twice?)")
"""),
("md", "## 2 · Distributions\nSkewed inputs (viscosity, permeability) span several orders of magnitude: a model sees almost all points squeezed near zero."),
("code", """
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
eor.viscosity_cp.plot.hist(bins=40, ax=ax[0], title="viscosity (cP)")
np.log10(eor.viscosity_cp.clip(lower=0.1)).plot.hist(bins=40, ax=ax[1], title="log10 viscosity")
eor.eor_method.value_counts().plot.bar(ax=ax[2], title="EOR method (class balance)", rot=30)
plt.tight_layout(); plt.show()
"""),
("md", "## 3 · Outliers with the IQR rule\nThe course function `detect_outliers_iqr` flags values outside Q1 − 1.5·IQR and Q3 + 1.5·IQR."),
("code", """
outlier_table(asph)
"""),
("code", """
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
plot_outliers(asph, ASPH_TARGET, ax=ax[0])
for (T, C), g in asph.groupby(["temperature_K", "alkane_carbon_number"]):
    ax[1].plot(g.dilution_ratio, g[ASPH_TARGET], "o-", ms=3, alpha=.8, color="tab:red" if T == 332 and C <= 8 else "tab:grey")
ax[1].set_xlabel("dilution ratio (mL/g)"); ax[1].set_ylabel("wt% precipitated"); ax[1].set_title("each line = one temperature + n-alkane series")
plt.tight_layout(); plt.show()
"""),
("md", """
The IQR rule flags 15 values. Plotted by series (right), they are not random spikes: they belong to **three complete
series** of 17 tests (332 K with n-C6, C7, C8) that are about ten times higher than every other series, while 332 K with C9–C12 looks normal. A statistical rule finds
them; only engineering judgement can decide whether they are a different unit, a different oil or a real effect.

`load_asphaltene()` keeps them and adds a `suspect` column so you can model with and without them (Lab 1).
"""),
("md", """
### ✏️ Your turn 2
Run `detect_outliers_iqr` on every numeric column of `vfm`. Are any of the flagged values *errors*? (Hint: look at the
water rates and remember that water cut goes from 0 to 0.8 in the tests.)
"""),
("ex", """
# loop over the numeric columns of vfm and print how many outliers each has
""", """
print(outlier_table(vfm))
print("\\nNo column has IQR outliers: wide ranges by design (10 chokes, water cut 0-0.8), not errors.")
"""),
("md", "## 4 · A hidden problem: duplicated tests\nThe VFM file looks large (2,100 rows). How many *different* well tests does it hold?"),
("code", """
phys = ["choke_in", "bhp_psi", "whp_psi", "wht_F", "tsep_F", "oil_rate_stb_d", "water_rate_bbl_d", "gas_rate_mmscf_d"]
print("rows:", len(vfm), " different physical tests:", len(vfm[phys].drop_duplicates()))
vfm[(vfm.bhp_psi == 5410.33)][["choke_in", "bhp_psi", "whp_psi", "psep_psi", "oil_rate_stb_d"]]
"""),
("md", """
The same well test is repeated for each separator pressure (100–600 psi), and the rates do not change. With a random
train/test split the "test" rows are copies of training rows, so any model looks perfect. `load_vfm()` adds a `case_id`;
always split by it (Lab 3).

## 5 · Scaling and splitting
Fit the scaler on the **training** rows only, then apply it to the test rows.
"""),
("code", """
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X, y = asph[ASPH_FEATURES], asph[ASPH_TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler().fit(X_train)
print(pd.DataFrame(sc.transform(X_train), columns=ASPH_FEATURES).describe().loc[["mean", "std"]].round(3))
mm = MinMaxScaler().fit(X_train); print(pd.DataFrame(mm.transform(X_test), columns=ASPH_FEATURES).describe().loc[["min", "max"]].round(3))
"""),
("md", """
### ✏️ Your turn 3
For the EOR data, make a **stratified** 80/20 split (`stratify=y`) and print the class counts of the test set.
Why is stratification important here? (Hint: how many Nitrogen projects are there?)
"""),
("ex", """
y = eor.eor_method
# X_train, X_test, y_train, y_test = train_test_split(...)
""", """
y = eor.eor_method
X_train, X_test, y_train, y_test = train_test_split(eor[EOR_FEATURES], y, test_size=0.2, random_state=42, stratify=y)
print(y_test.value_counts())
print("Only 7 Nitrogen projects in total: without stratify a test set can easily get 0 or 4 of them.")
"""),
("md", """
## Key ideas
* Read `describe()` like an engineer: a 0 cP oil or a 5 °F reservoir is a data error, not a feature.
* The IQR rule finds unusual values; deciding whether they are wrong needs domain knowledge.
* Duplicated rows make random splits leak: split by the physical case.
* Log-transform inputs that span orders of magnitude; scale using training data only; stratify small classes.
"""),
]

# =====================================================================================
LABS["Lab1_Regression_Asphaltene_Precipitation"] = [
("md", """
# Lab 1 · Regression: asphaltene precipitation
**Slides:** Module 06 (algorithms) and 07 (evaluation)  ·  **Time:** 75 min

Adding an n-alkane (n-C5 … n-C12) to crude oil precipitates asphaltene. We predict the **weight percent precipitated**
from temperature, the alkane carbon number and the dilution ratio (mL alkane per g oil), and compare ten algorithms,
the same ones as the course notebooks (`Linear_Regression`, `KNN`, `SVR`, `DT`, `RF`, `AdaBoost`, `Gradient_Boosting`, `NN`, `Voting`).

**You will:** compare algorithms · tune with cross-validation instead of the test set · test on a whole unseen series · explain the model.
"""),
("code", SETUP + """
from aiml.data import load_asphaltene, ASPH_FEATURES, ASPH_TARGET
from aiml.models import regressors
from aiml.evaluate import compare, crossplot, hyper_curve, report, importance
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupKFold, cross_val_predict
from sklearn.metrics import r2_score
df = load_asphaltene()
X, y = df[ASPH_FEATURES], df[ASPH_TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""),
("md", "## 1 · Ten algorithms on all 167 tests"),
("code", """
res_all = compare(regressors(), X_train, y_train, X_test, y_test); res_all[["model", "R2_train", "R2_test", "MAE_test", "fit_s"]]
"""),
("md", "## 2 · The same comparison without the 17 suspect tests (Lab 0)"),
("code", """
clean = load_asphaltene(drop_suspect=True)
Xc, yc = clean[ASPH_FEATURES], clean[ASPH_TARGET]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)
res_clean = compare(regressors(), Xc_train, yc_train, Xc_test, yc_test)
res_all[["model", "R2_test"]].merge(res_clean[["model", "R2_test"]], on="model", suffixes=("_all", "_clean"))
"""),
("md", """
Seventeen rows decide the ranking. With them, only flexible models (trees, neural nets) can chase the three extreme
series and linear regression fails; without them, even linear regression reaches R² ≈ 0.96. **Clean data first,
then choose the algorithm.** We continue with the clean data.

## 3 · Tuning: the course loops, done safely
The course notebooks loop over `max_depth` or `n_neighbors` and pick the value with the best **test** score. That uses
the test set to make a decision, so the test score is no longer an honest estimate. Use cross-validation on the
training data to tune, and touch the test set once at the end.
"""),
("code", """
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
hyper_curve(RandomForestRegressor(n_estimators=100, random_state=0), "max_depth", list(range(1, 15)), Xc_train, yc_train, Xc_test, yc_test, ax=ax[0])
cv = KFold(5, shuffle=True, random_state=0)
ks = list(range(1, 16))
cv_scores = [cross_val_score(make_pipeline(StandardScaler(), KNeighborsRegressor(k)), Xc_train, yc_train, cv=cv).mean() for k in ks]
ax[1].plot(ks, cv_scores, "o-"); ax[1].set_xlabel("n_neighbors"); ax[1].set_ylabel("5-fold CV R² (training data only)")
ax[1].set_title(f"KNN: best k by cross-validation = {ks[int(np.argmax(cv_scores))]}"); plt.tight_layout(); plt.show()
"""),
("md", """
### ✏️ Your turn 1
Tune `C` of the SVR (`regressors()["SVR"]`, parameter name `svr__C`) over `[0.3, 1, 3, 10, 30, 100, 300]` with 5-fold
cross-validation on the training data. Then fit the best one and report the test score once.
"""),
("ex", """
svr = regressors()["SVR"]
Cs = [0.3, 1, 3, 10, 30, 100, 300]
# cross_val_score(svr.set_params(svr__C=c), Xc_train, yc_train, cv=cv).mean()
""", """
svr = regressors()["SVR"]
Cs = [0.3, 1, 3, 10, 30, 100, 300]
s = [cross_val_score(svr.set_params(svr__C=c), Xc_train, yc_train, cv=cv).mean() for c in Cs]
best = Cs[int(np.argmax(s))]; print(dict(zip(Cs, np.round(s, 3))), "→ best C =", best)
svr.set_params(svr__C=best).fit(Xc_train, yc_train); report(yc_test, svr.predict(Xc_test), "SVR test (once)")
"""),
("md", """
## 4 · A harder, more honest test: unseen series
A random split puts points of the same curve in train and test, so the model only interpolates along a curve it
already saw. In practice you want to predict a **new temperature or a new precipitant**. `GroupKFold` holds out whole
series (one temperature + n-alkane curve at a time).
"""),
("code", """
groups = clean["series"]
rows = []
for name, m in regressors().items():
    p_rand = cross_val_predict(m, Xc, yc, cv=KFold(5, shuffle=True, random_state=0))
    p_grp = cross_val_predict(m, Xc, yc, cv=GroupKFold(5), groups=groups)
    rows.append({"model": name, "random 5-fold R²": r2_score(yc, p_rand), "unseen-series R²": r2_score(yc, p_grp)})
pd.DataFrame(rows).sort_values("unseen-series R²", ascending=False).round(3)
"""),
("md", """
### ✏️ Your turn 2 · physics features
Precipitation grows roughly with the **logarithm** of dilution. Add `log_dilution` (`load_asphaltene(drop_suspect=True, add_features=True)`)
and repeat the unseen-series test for *Linear regression*, *KNN* and *Neural network (MLP)*. Which model gains most, and why?
"""),
("ex", """
cf = load_asphaltene(drop_suspect=True, add_features=True)
F2 = ASPH_FEATURES + ["log_dilution"]
""", """
cf = load_asphaltene(drop_suspect=True, add_features=True)
F2 = ASPH_FEATURES + ["log_dilution"]
for name in ["Linear regression", "KNN", "Neural network (MLP)"]:
    m = regressors()[name]
    a = r2_score(yc, cross_val_predict(m, Xc, yc, cv=GroupKFold(5), groups=groups))
    b = r2_score(cf[ASPH_TARGET], cross_val_predict(m, cf[F2], cf[ASPH_TARGET], cv=GroupKFold(5), groups=cf["series"]))
    print(f"{name:<24} without {a:.3f}  with log_dilution {b:.3f}")
print("Linear regression gains most: a straight line cannot bend, the log feature bends it for the model.")
"""),
("md", "## 5 · Final model, cross-plot and explanation"),
("code", """
best = regressors()["Gradient boosting"].fit(Xc_train, yc_train)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
crossplot(yc_train, best.predict(Xc_train), yc_test, best.predict(Xc_test), "Gradient boosting · asphaltene", "wt%", ax=ax[0])
imp = importance(best, Xc_test, yc_test, ax=ax[1]); ax[1].set_title("permutation importance (test data)"); plt.tight_layout(); plt.show()
"""),
("code", """
# what-if: precipitation curves the model predicts for n-heptane at four temperatures
dr = np.linspace(3, 25, 60)
for T in [293, 308, 323, 332]:
    plt.plot(dr, best.predict(pd.DataFrame({"temperature_K": T, "alkane_carbon_number": 7, "dilution_ratio": dr})), label=f"{T} K")
plt.xlabel("dilution ratio (mL/g)"); plt.ylabel("predicted wt%"); plt.title("model response · n-C7"); plt.legend(); plt.show()
"""),
("md", """
### ✏️ Your turn 3
Look at the 332 K curve. Is it smooth and physical between the points the model saw? Tree ensembles predict in **steps**;
try `regressors()["Neural network (MLP)"]` or `"SVR"` and compare the shape. Which would you trust for a new dilution ratio?

## Key ideas
* Data quality (17 rows) changed the winner more than any algorithm choice.
* Tune with cross-validation on training data; use the test set once.
* Split the way the model will be used: unseen series is harder and more honest than a random split.
* A simple physics feature (log dilution) can make a simple model as good as a complex one.
"""),
("ex", "# compare the shape of the response curves for another model\n", """
for name in ["Neural network (MLP)", "SVR"]:
    m = regressors()[name].fit(Xc_train, yc_train)
    for T in [293, 332]:
        plt.plot(dr, m.predict(pd.DataFrame({"temperature_K": T, "alkane_carbon_number": 7, "dilution_ratio": dr})), label=f"{name} · {T} K")
plt.xlabel("dilution ratio"); plt.ylabel("predicted wt%"); plt.legend(); plt.show()
"""),
]

# =====================================================================================
LABS["Lab2_Classification_EOR_Screening"] = [
("md", """
# Lab 2 · Classification: EOR screening
**Slides:** Module 06 and 07 (classification metrics, ROC)  ·  **Time:** 75 min

455 field projects worldwide, each with the EOR method that was applied: CO₂, hydrocarbon or nitrogen gas injection,
polymer flooding, in-situ combustion or steam. From six reservoir properties we predict the method, then compare the
machine-learning screen with the classic **Taber screening rules**.

**You will:** handle imbalanced classes · compare classifiers · read confusion matrices · understand micro, macro and weighted averages · screen a new field.
"""),
("code", SETUP + """
from aiml.data import load_eor, EOR_FEATURES, EOR_CLASSES
from aiml.models import classifiers
from aiml.evaluate import compare, confusion, classification_metrics, importance
from aiml.generators import taber_screen, TABER
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
df = load_eor(log_viscosity=True)
print(df.eor_method.value_counts())
"""),
("md", "## 1 · Inputs: raw or log?\nViscosity runs from 0.1 to 5,000,000 cP. Trees do not care about scale; distance-based models (KNN, SVC, MLP) do."),
("code", """
F_RAW = EOR_FEATURES
F_LOG = ["log_permeability", "depth_ft", "api_gravity", "log_viscosity", "temperature_F", "oil_saturation_pct"]
y = df.eor_method
tr, te = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
for name in ["KNN", "SVC", "Random forest"]:
    m = classifiers()[name]
    a = m.fit(df.loc[tr, F_RAW], y[tr]).score(df.loc[te, F_RAW], y[te]); b = m.fit(df.loc[tr, F_LOG], y[tr]).score(df.loc[te, F_LOG], y[te])
    print(f"{name:<14} raw inputs {a:.3f}   log inputs {b:.3f}")
"""),
("md", "## 2 · Compare classifiers\nWith imbalanced classes, accuracy hides the small classes; sort by **macro F1** (every class counts the same)."),
("code", """
X_train, X_test, y_train, y_test = df.loc[tr, F_LOG], df.loc[te, F_LOG], y[tr], y[te]
res = compare(classifiers(), X_train, y_train, X_test, y_test, task="classification")
res[["model", "acc_train", "accuracy_test", "F1_macro_test", "F1_micro_test", "F1_weighted_test", "ROC_AUC_ovr_test"]]
"""),
("md", """
**Micro** averaging pools all predictions (equals accuracy for single-label problems). **Macro** averages the per-class
scores, so Nitrogen (7 projects) counts as much as Steam (197). **Weighted** averages by class size. The course
figures `Micro_Average.JPG`, `Macro_Average.JPG` and `Weighted_Average.JPG` show the same three numbers.
"""),
("code", """
rf = classifiers()["Random forest"].fit(X_train, y_train); p = rf.predict(X_test)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
confusion(y_test, p, EOR_CLASSES, ax=ax[0], title="Random forest · test counts")
confusion(y_test, p, EOR_CLASSES, ax=ax[1], title="row-normalized (recall per class)", normalize=True)
plt.tight_layout(); plt.show()
from sklearn.metrics import classification_report; print(classification_report(y_test, p, zero_division=0))
"""),
("md", """
### ✏️ Your turn 1 · cross-validated comparison
One 20% test set holds only ~2 Nitrogen projects, so its scores are noisy. Compare *Random forest*, *Gradient boosting*,
*SVC* and *Neural network (MLP)* with **stratified 5-fold cross-validation** (`scoring="f1_macro"`) on all data. Report mean ± std.
"""),
("ex", """
cv = StratifiedKFold(5, shuffle=True, random_state=0)
""", """
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name in ["Random forest", "Gradient boosting", "SVC", "Neural network (MLP)"]:
    s = cross_val_score(classifiers()[name], df[F_LOG], y, cv=cv, scoring="f1_macro")
    print(f"{name:<22} macro F1 {s.mean():.3f} ± {s.std():.3f}")
"""),
("md", "## 3 · Machine learning vs the expert rules\n`taber_screen` applies simplified Taber, Martin & Seright (1997) criteria, the rules an engineer would use."),
("code", """
pd.DataFrame(TABER).T
"""),
("code", """
t = taber_screen(df.loc[te])
print("Taber rules  accuracy", round((t == y_test).mean(), 3), " macro F1", round(classification_metrics(y_test, t)["F1_macro"], 3))
print("Random forest accuracy", round((p == y_test).mean(), 3), " macro F1", round(classification_metrics(y_test, p)["F1_macro"], 3))
fig, ax = plt.subplots(figsize=(7, 5.5)); confusion(y_test, t, EOR_CLASSES, ax=ax, title="Taber rules · test"); plt.show()
"""),
("md", """
The rules know nothing about this database, yet get most projects right; the model learns the operators' real
choices (for example, which light-oil reservoirs got CO₂ rather than hydrocarbon gas). Rules are transparent and work
outside the data; the model is more accurate inside it. Use both.

## 4 · Explain and screen a new field
"""),
("code", """
imp = importance(rf, X_test, y_test); plt.title("permutation importance (accuracy drop)"); plt.show()
"""),
("code", """
new = pd.DataFrame([{"permeability_md": 100, "depth_ft": 8000, "api_gravity": 30, "viscosity_cp": 10, "temperature_F": 150, "oil_saturation_pct": 50}])
new_log = new.assign(log_permeability=np.log10(new.permeability_md), log_viscosity=np.log10(new.viscosity_cp))[F_LOG]
print("model:", rf.predict(new_log)[0], "  rules:", taber_screen(new)[0])
pd.Series(rf.predict_proba(new_log)[0], rf.classes_).sort_values(ascending=False).round(3)
"""),
("md", """
### ✏️ Your turn 2
A heavy-oil field: permeability 1500 mD, depth 1200 ft, 12 °API, 3000 cP, 95 °F, 70 % oil saturation.
What do the model and the rules recommend, and how confident is the model? Then make the same field 6000 ft deep. What changes?
"""),
("ex", """
heavy = pd.DataFrame([{"permeability_md": 1500, "depth_ft": 1200, "api_gravity": 12, "viscosity_cp": 3000, "temperature_F": 95, "oil_saturation_pct": 70}])
""", """
def screen(f):
    fl = f.assign(log_permeability=np.log10(f.permeability_md), log_viscosity=np.log10(f.viscosity_cp))[F_LOG]
    pr = pd.Series(rf.predict_proba(fl)[0], rf.classes_).sort_values(ascending=False).round(2)
    print("model:", pr.index[0], dict(pr.head(3)), "| rules:", taber_screen(f)[0])
heavy = pd.DataFrame([{"permeability_md": 1500, "depth_ft": 1200, "api_gravity": 12, "viscosity_cp": 3000, "temperature_F": 95, "oil_saturation_pct": 70}])
screen(heavy); screen(heavy.assign(depth_ft=6000))
print("At 6000 ft steam loses too much heat in the wellbore, so the rules switch to combustion.")
print("The model only follows if the database holds deep heavy-oil projects: check its probabilities.")
"""),
("md", """
### ✏️ Your turn 3 · class weights
Retrain the random forest with `class_weight="balanced"` and compare recall for *Nitrogen*, *Combustion* and *Polymer*.

## Key ideas
* Log-transform inputs that span decades before distance-based models.
* With imbalanced classes report macro F1 and the confusion matrix, not just accuracy.
* Cross-validate when the test set is small.
* Expert rules are a strong, explainable baseline; ML should beat them and agree with them where it matters.
"""),
("ex", "# RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=0)\n", """
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
rb = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0).fit(X_train, y_train)
for name, m in [("plain", rf), ("balanced", rb)]:
    r = recall_score(y_test, m.predict(X_test), labels=EOR_CLASSES, average=None, zero_division=0)
    print(name, dict(zip(EOR_CLASSES, r.round(2))))
"""),
]

# =====================================================================================
LABS["Lab3_Virtual_Flow_Metering"] = [
("md", """
# Lab 3 · Virtual flow metering (VFM)
**Slides:** Module 03 (production applications), 06 and 07  ·  **Time:** 75 min

A test separator measures oil, water and gas only occasionally. A **virtual flow meter** estimates the rates
continuously from gauges the well already has: choke size, bottom-hole and wellhead pressure, wellhead temperature,
separator pressure and temperature. The course notebook `VFM.ipynb` compares XGBoost, Extra Trees, CatBoost, random
forest and a neural network.

**You will:** find and fix a leakage problem · compare models with a group split · predict three phases · test extrapolation to an unseen choke · run a what-if.
"""),
("code", SETUP + """
from aiml.data import load_vfm, VFM_FEATURES, VFM_TARGETS
from aiml.models import regressors
from aiml.evaluate import compare, crossplot, report, importance
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, KFold, cross_val_score
from sklearn.metrics import r2_score
df = load_vfm()
print(len(df), "rows ·", df.case_id.nunique(), "different well tests"); df.head()
"""),
("md", "## 1 · Why the course notebook scores R² ≈ 1.000\nEach test appears once per separator pressure. Compare a random split with a split by `case_id`."),
("code", """
from sklearn.ensemble import ExtraTreesRegressor
X, y = df[VFM_FEATURES], df["oil_rate_stb_d"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=123)
et = ExtraTreesRegressor(n_estimators=300, random_state=0).fit(Xtr, ytr); report(yte, et.predict(Xte), "random split")
g = [c for c in VFM_FEATURES if c != "psep_psi"]
print("test rows whose well test is also in training (only Psep differs):", Xte[g].merge(Xtr[g].drop_duplicates(), how="inner").shape[0], "of", len(Xte))
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
tr, te = next(gss.split(X, y, groups=df.case_id))
et.fit(X.iloc[tr], y.iloc[tr]); report(y.iloc[te], et.predict(X.iloc[te]), "split by case_id")
"""),
("md", "The drop is small here because the tests are dense, but the random-split number was not an honest estimate. From now on: **group split**."),
("code", """
X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
keep = ["Linear regression", "KNN", "Decision tree", "Random forest", "Extra trees", "Gradient boosting", "Hist. gradient boosting (XGBoost-style)", "Neural network (MLP)"]
res = compare({k: v for k, v in regressors().items() if k in keep}, X_train, y_train, X_test, y_test)
res[["model", "R2_train", "R2_test", "MAE_test", "MAPE_%_test", "fit_s"]]
"""),
("md", """
### ✏️ Your turn 1 · 10-fold cross-validation, done right
The notebook uses `KFold(10, shuffle=True)`. Repeat it for *Extra trees* with `KFold` and with `GroupKFold(10)` (groups = `case_id`). Which is optimistic?
"""),
("ex", """
m = regressors()["Extra trees"]
""", """
m = regressors()["Extra trees"]
a = cross_val_score(m, X, y, cv=KFold(10, shuffle=True, random_state=123), scoring="r2")
b = cross_val_score(m, X, y, cv=GroupKFold(10), groups=df.case_id, scoring="r2")
print(f"KFold      R² {a.mean():.4f} ± {a.std():.4f}"); print(f"GroupKFold R² {b.mean():.4f} ± {b.std():.4f}  (honest)")
"""),
("md", "## 2 · Cross-plots and residuals"),
("code", """
best = regressors()["Extra trees"].fit(X_train, y_train)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
crossplot(y_train, best.predict(X_train), y_test, best.predict(X_test), "Extra trees · oil rate", "STB/d", ax=ax[0])
ax[1].scatter(y_test, y_test - best.predict(X_test), s=12, alpha=.6); ax[1].axhline(0, color="r", ls="--")
ax[1].set_xlabel("actual oil rate"); ax[1].set_ylabel("residual"); ax[1].set_title("residuals (test)"); plt.tight_layout(); plt.show()
"""),
("md", "## 3 · Three phases\nOne model per phase. Which phase is hardest to meter virtually?"),
("code", """
rows = []
for t in VFM_TARGETS:
    m = regressors()["Extra trees"].fit(X_train, df[t].iloc[tr]); p = m.predict(X_test)
    rows.append({"target": t, **{k: round(v, 4) for k, v in report(df[t].iloc[te], p, t).items()}})
pd.DataFrame(rows)
"""),
("md", """
## 4 · Extrapolation: a choke the model never saw
Hold out every test at one choke size and train on the other nine. Interpolating between chokes (0.70 in) is very different
from going beyond the data, and tree models cannot predict outside the range they saw.
"""),
("code", """
rows = []
for c in [0.25, 0.70, 1.94, 2.50]:
    m_tr = df.choke_in != c
    for name in ["Extra trees", "Hist. gradient boosting (XGBoost-style)", "Neural network (MLP)", "Linear regression"]:
        m = regressors()[name].fit(X[m_tr], y[m_tr])
        rows.append({"held-out choke (in)": c, "model": name, "R2": round(r2_score(y[~m_tr], m.predict(X[~m_tr])), 3)})
pd.DataFrame(rows).pivot(index="model", columns="held-out choke (in)", values="R2")
"""),
("md", """
### ✏️ Your turn 2 · what-if
Keep a mid-life well (BHP 4700 psi, WHT 150 °F, Psep 300 psi, Tsep 100 °F) and vary the choke from 0.25 to 2.5 in.
Feed WHP from the test closest to each choke and plot the predicted oil rate. Why does the curve flatten above ~1.2 in?
"""),
("ex", """
chokes = np.sort(df.choke_in.unique())
""", """
chokes = np.sort(df.choke_in.unique())
whp = df.groupby("choke_in").whp_psi.median()
q = best.predict(pd.DataFrame({"choke_in": chokes, "bhp_psi": 4700, "whp_psi": whp[chokes].values, "wht_F": 150, "psep_psi": 300, "tsep_F": 100}))
plt.plot(chokes, q, "o-"); plt.xlabel("choke (in)"); plt.ylabel("predicted oil rate (STB/d)"); plt.title("what-if: choke opening"); plt.show()
print("Above ~1.2 in the choke stops controlling the flow: the reservoir and tubing limit the rate (subcritical flow).")
"""),
("md", "## 5 · Which gauges matter?"),
("code", """
importance(best, X_test, y_test); plt.title("permutation importance · oil rate"); plt.show()
"""),
("md", """
### ✏️ Your turn 3 · a failed gauge
The wellhead pressure gauge fails. Retrain without `whp_psi` and compare the test R². Which other gauge takes over?

## Key ideas
* Duplicated rows make a random split leak: always ask "how many independent tests do I have?".
* VFM interpolates well; extrapolation to unseen operating conditions needs physics or new tests.
* On this data the six gauges predict all three phases well (test R² about 0.99 with a group split).
* An unseen choke size is the real weakness: the meter is reliable only inside the conditions it was trained on.
* Permutation importance shows which gauges the meter depends on, and what a failed gauge costs.
"""),
("ex", "# retrain without whp_psi\n", """
F2 = [f for f in VFM_FEATURES if f != "whp_psi"]
m2 = regressors()["Extra trees"].fit(X_train[F2], y_train); report(y_test, m2.predict(X_test[F2]), "without WHP")
importance(m2, X_test[F2], y_test); plt.show()
"""),
]

# =====================================================================================
LABS["Lab4_Bias_Variance_and_Explainability"] = [
("md", """
# Lab 4 · Bias, variance, regularization and explainability
**Slides:** Module 07 (bias and variance, learning curve, regularization, explainable AI) and Module 06 (clustering)  ·  **Time:** 60 min

**You will:** see under- and over-fitting · draw learning curves · regularize a neural network · read partial-dependence plots · cluster fields without labels.
"""),
("code", SETUP + """
from aiml.data import load_asphaltene, load_eor, load_vfm, ASPH_FEATURES, ASPH_TARGET, VFM_FEATURES
from aiml.models import regressors
from sklearn.model_selection import learning_curve, validation_curve, KFold, GroupKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
a = load_asphaltene(drop_suspect=True)
"""),
("md", "## 1 · Under- and over-fitting on one curve\nFit polynomials of growing degree to one precipitation series (n-C7, 308 K)."),
("code", """
s = a[(a.temperature_K == 308) & (a.alkane_carbon_number == 7)]
x, yv = s[["dilution_ratio"]].values, s[ASPH_TARGET].values; xx = np.linspace(x.min(), x.max(), 200)[:, None]
for d in [1, 2, 4]:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x, yv); plt.plot(xx, m.predict(xx), label=f"degree {d}")
plt.scatter(x, yv, c="k", zorder=3, label="data"); plt.ylim(0, yv.max() * 1.6); plt.legend(); plt.xlabel("dilution ratio"); plt.ylabel("wt%"); plt.show()
"""),
("md", "## 2 · Validation curve: model complexity"),
("code", """
X, y = a[ASPH_FEATURES].astype(float), a[ASPH_TARGET]; cv = GroupKFold(5); g = a.series
depths = list(range(1, 13))
tr, va = validation_curve(DecisionTreeRegressor(random_state=0), X, y, param_name="max_depth", param_range=depths, cv=cv, groups=g, scoring="r2")
plt.plot(depths, tr.mean(1), "o-", label="train"); plt.plot(depths, va.mean(1), "s-", label="validation (unseen series)")
plt.xlabel("max_depth"); plt.ylabel("R²"); plt.legend(); plt.title("bias ← → variance"); plt.show()
"""),
("md", "## 3 · Learning curve: would more data help?\nVFM data, group 5-fold, extra trees vs linear regression."),
("code", """
v = load_vfm()
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for axi, name in zip(ax, ["Linear regression", "Extra trees"]):
    n, tr, va = learning_curve(regressors()[name], v[VFM_FEATURES], v.oil_rate_stb_d, cv=GroupKFold(5), groups=v.case_id, train_sizes=np.linspace(.1, 1, 6), scoring="r2")
    axi.plot(n, tr.mean(1), "o-", label="train"); axi.plot(n, va.mean(1), "s-", label="validation"); axi.set_title(name); axi.set_xlabel("training rows"); axi.legend()
plt.tight_layout(); plt.show()
"""),
("md", """
Linear regression: train and validation meet early at a low score → **high bias**, more data will not help, a better model will.
Extra trees: a gap that shrinks with data → **variance**, more tests still help.

### ✏️ Your turn 1 · regularization
Fit `MLPRegressor(hidden_layer_sizes=(64, 64), solver="lbfgs", max_iter=5000)` on the asphaltene data with
`alpha` in `[1e-5, 1e-3, 1e-1, 1, 10]` and draw the group-CV validation curve (parameter `mlpregressor__alpha`).
"""),
("ex", """
nn = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), solver="lbfgs", max_iter=5000, random_state=0))
alphas = [1e-5, 1e-3, 1e-1, 1, 10]
""", """
nn = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), solver="lbfgs", max_iter=5000, random_state=0))
alphas = [1e-5, 1e-3, 1e-1, 1, 10]
tr, va = validation_curve(nn, X, y, param_name="mlpregressor__alpha", param_range=alphas, cv=cv, groups=g, scoring="r2")
plt.semilogx(alphas, tr.mean(1), "o-", label="train"); plt.semilogx(alphas, va.mean(1), "s-", label="validation"); plt.xlabel("alpha (L2 penalty)"); plt.legend(); plt.show()
"""),
("md", "## 4 · Explainable AI: partial dependence\nHow does the model's prediction change with one input, averaged over the others?"),
("code", """
from sklearn.inspection import PartialDependenceDisplay
gb = regressors()["Gradient boosting"].fit(X, y)
PartialDependenceDisplay.from_estimator(gb, X, ASPH_FEATURES, kind="both", subsample=60, random_state=0)
plt.gcf().set_size_inches(14, 4); plt.tight_layout(); plt.show()
"""),
("md", """
Each panel is an engineering check: precipitation should **rise with dilution**, **fall with carbon number** (heavier
alkanes are better solvents) and fall with temperature. A model that is accurate but violates physics is not ready.
The course notebooks use **SHAP** for the same purpose (`pip install shap`): `shap.Explainer(model.predict, X)`.

## 5 · Unsupervised: clustering EOR projects without labels
"""),
("code", """
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
e = load_eor(log_viscosity=True); F = ["log_viscosity", "api_gravity", "depth_ft", "log_permeability"]
Z = StandardScaler().fit_transform(e[F])
ks = range(2, 9); sil = [silhouette_score(Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z)) for k in ks]
plt.plot(ks, sil, "o-"); plt.xlabel("k"); plt.ylabel("silhouette"); plt.show()
e["cluster"] = KMeans(3, n_init=10, random_state=0).fit_predict(Z)
pd.crosstab(e.cluster, e.eor_method)
"""),
("md", """
### ✏️ Your turn 2
Describe each of the three clusters in engineering terms (median API, viscosity, depth). Do the clusters match
"thermal", "gas" and "chemical" EOR?

## Key ideas
* Validation curves show complexity; learning curves show whether data or model is the bottleneck.
* Regularization (alpha, depth, number of neighbours) is the dial between bias and variance.
* Partial dependence and SHAP turn a black box into curves an engineer can check against physics.
* Clustering finds structure without labels; labels then tell you what the clusters mean.
"""),
("ex", "# e.groupby('cluster')[...].median()\n", """
print(e.groupby("cluster")[["api_gravity", "viscosity_cp", "depth_ft", "permeability_md"]].median().round(0))
print("Typically: one shallow heavy-oil cluster (thermal), one deep light-oil cluster (gas), one in between (chemical / mixed).")
"""),
]

# =====================================================================================
LABS["Lab5_Generate_Your_Own_Data"] = [
("md", """
# Lab 5 · Generate your own data
**Time:** 60 min  ·  Code: `aiml/generators.py` (also from a terminal: `python -m aiml.generators vfm --n 1000`)

Real datasets are small and fixed. With a generator you control the physics, the noise and the sampling, so you can
ask questions like "how many tests do I need?" or "how much gauge noise can my VFM tolerate?".

| Generator | Physics inside |
|---|---|
| `make_asphaltene_data` | plateau(T, n-alkane) × (1 − exp(−(dilution − onset)/b)), fitted to the 150 clean course tests |
| `make_vfm_data` | inflow `q = PI (Pr − Pwf)` + tubing gradient + Gilbert choke `Pwh = 435 GLR^0.546 q / S^1.89` |
| `make_eor_data` | projects sampled class by class from the course database; optional Taber-rule labels |
"""),
("code", SETUP + """
from aiml.generators import make_asphaltene_data, make_vfm_data, make_eor_data, taber_screen
from aiml.data import load_vfm, load_eor, load_asphaltene, VFM_FEATURES, EOR_FEATURES, ASPH_FEATURES, ASPH_TARGET
from aiml.models import regressors, classifiers
from aiml.evaluate import report
from sklearn.metrics import r2_score
"""),
("md", "## 1 · Synthetic well tests"),
("code", """
syn = make_vfm_data(n_tests=800, noise=0.01, seed=0)
real = load_vfm()
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for d, lab in [(real, "course data"), (syn, "generator")]:
    ax[0].scatter(d.choke_in, d.oil_rate_stb_d + d.water_rate_bbl_d, s=8, alpha=.4, label=lab)
    ax[1].scatter(d.whp_psi, d.oil_rate_stb_d + d.water_rate_bbl_d, s=8, alpha=.4, label=lab)
ax[0].set_xlabel("choke (in)"); ax[1].set_xlabel("WHP (psi)"); ax[0].set_ylabel("liquid rate (bbl/d)"); ax[0].legend(); plt.show()
"""),
("md", "## 2 · How much gauge noise can a VFM tolerate?\nEven with perfect gauges R² stays below 1: the oil rate needs the **water cut**, and no gauge measures it. (The course data has only five water cuts, 0 to 0.8, which makes it easier.)"),
("code", """
rows = []
for noise in [0, 0.005, 0.01, 0.02, 0.05]:
    d = make_vfm_data(800, noise=noise, seed=1); t = make_vfm_data(300, noise=noise, seed=2)
    m = regressors()["Extra trees"].fit(d[VFM_FEATURES], d.oil_rate_stb_d)
    rows.append({"gauge noise": noise, "test R2": round(r2_score(t.oil_rate_stb_d, m.predict(t[VFM_FEATURES])), 4)})
pd.DataFrame(rows)
"""),
("md", """
### ✏️ Your turn 1 · how many tests do I need?
With `noise=0.01`, train extra trees on 50, 100, 200, 400 and 800 synthetic tests and test on 300 fresh ones. Plot R² vs number of tests.
"""),
("ex", """
test = make_vfm_data(300, noise=0.01, seed=99)
""", """
test = make_vfm_data(300, noise=0.01, seed=99); ns = [50, 100, 200, 400, 800]; r = []
for n in ns:
    d = make_vfm_data(n, noise=0.01, seed=n); m = regressors()["Extra trees"].fit(d[VFM_FEATURES], d.oil_rate_stb_d)
    r.append(r2_score(test.oil_rate_stb_d, m.predict(test[VFM_FEATURES])))
plt.semilogx(ns, r, "o-"); plt.xlabel("well tests"); plt.ylabel("test R²"); plt.show()
"""),
("md", "## 3 · Train on synthetic, test on real (sim-to-real)\nA model trained only on the generator, tested on the course well: how far is our physics from the data?"),
("code", """
m = regressors()["Extra trees"].fit(syn[VFM_FEATURES], syn.oil_rate_stb_d + syn.water_rate_bbl_d)
report(real.liquid_rate_bbl_d, m.predict(real[VFM_FEATURES]), "synthetic → real (liquid)")
"""),
("md", "## 4 · Asphaltene with planted outliers\nThe generator can plant outliers (here: 6 % of tests multiplied by 1.6–2.5), so you know exactly which ones a method should find."),
("code", """
from aiml.prep import detect_outliers_iqr
d = make_asphaltene_data(noise=0.05, outliers=0.06, outlier_factor=(1.6, 2.5), seed=3)
lo, hi, idx = detect_outliers_iqr(d, ASPH_TARGET)
found = d.index.isin(idx)
print("planted:", d.is_outlier.sum(), " flagged by IQR:", found.sum(), " correctly flagged:", (found & d.is_outlier).sum())
"""),
("md", """
### ✏️ Your turn 2
The IQR rule looks at the target alone, so an outlier on a low curve hides inside the normal range of the high curves.
Flag outliers by their **residual from a model** instead: predict every row with cross-validation (so a row never
predicts itself) and flag rows more than 35 % above the prediction. How many planted outliers do you find now?
"""),
("ex", """
# idea: fit a model, flag rows whose |residual| is large
""", """
from sklearn.model_selection import cross_val_predict, KFold
m = regressors()["Polynomial (deg 2)"]
p = cross_val_predict(m, d[ASPH_FEATURES], d[ASPH_TARGET], cv=KFold(5, shuffle=True, random_state=0))
res = (d[ASPH_TARGET] - p) / np.clip(p, 0.3, None)
flag = res > 0.35
print("planted:", d.is_outlier.sum(), " flagged:", flag.sum(), " correct:", (flag & d.is_outlier).sum())
"""),
("md", "## 5 · EOR: rules vs data labels"),
("code", """
syn_e = make_eor_data(n=1000, seed=0)
real_e = load_eor()
m = classifiers()["Random forest"].fit(syn_e[EOR_FEATURES], syn_e.eor_method)
print("trained on synthetic, accuracy on the real projects:", round(m.score(real_e[EOR_FEATURES], real_e.eor_method), 3))
print("Taber rules on the real projects:", round((taber_screen(real_e) == real_e.eor_method).mean(), 3))
"""),
("md", """
### ✏️ Your turn 3 (project)
Choose one generator, design a dataset that answers a question you care about (sampling, noise, size), train two
models, and write three sentences: what you asked, what you found, and what it means for a real field study.

## Key ideas
* A generator turns "is my model good?" into controlled experiments: data size, noise, outliers, extrapolation.
* Known truth lets you test methods (outlier detection, splits) before trusting them on field data.
* Sim-to-real gaps show where your physics or your data is incomplete.
"""),
("ex", "# your project\n", "# open-ended: see the examples above\n"),
]

if __name__ == "__main__":
    os.makedirs("../notebooks", exist_ok=True); os.makedirs("../solutions", exist_ok=True)
    for name, cells in LABS.items():
        if len(sys.argv) > 1 and not any(a in name for a in sys.argv[1:]): continue
        stu, sol = build(cells, name.replace("_", " "))
        nbf.write(stu, f"../notebooks/{name}.ipynb"); nbf.write(sol, f"../solutions/{name}_SOLUTION.ipynb")
        print("wrote", name)
