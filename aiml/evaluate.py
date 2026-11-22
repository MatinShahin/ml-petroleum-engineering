"""Metrics and plots of Module 07: cross-plots, hyper-parameter curves, confusion matrices, model comparison."""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics as met
from sklearn.base import clone
from sklearn.inspection import permutation_importance


def regression_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    nz = np.abs(y) > 1e-9
    return {"R2": met.r2_score(y, p), "MAE": met.mean_absolute_error(y, p), "RMSE": met.mean_squared_error(y, p) ** 0.5,
            "MAPE_%": 100 * np.mean(np.abs((y[nz] - p[nz]) / y[nz])) if nz.any() else np.nan}


def classification_metrics(y, p, proba=None, classes=None):
    out = {"accuracy": met.accuracy_score(y, p), "precision_macro": met.precision_score(y, p, average="macro", zero_division=0),
           "recall_macro": met.recall_score(y, p, average="macro", zero_division=0), "F1_macro": met.f1_score(y, p, average="macro", zero_division=0),
           "F1_micro": met.f1_score(y, p, average="micro"), "F1_weighted": met.f1_score(y, p, average="weighted", zero_division=0)}
    if proba is not None:
        try:
            out["ROC_AUC_ovr"] = met.roc_auc_score(y, proba, multi_class="ovr", labels=classes)
        except ValueError:
            pass
    return out


def report(y, p, name=""):
    m = regression_metrics(y, p)
    print(f"{name:<28} R2 {m['R2']:.3f}  MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}  MAPE {m['MAPE_%']:.1f}%")
    return m


def crossplot(y_train, p_train, y_test, p_test, title="", unit="", ax=None):
    """Predicted vs actual for train and test, with the y = x line and R2 of each."""
    ax = ax or plt.gca()
    lo = min(np.min(y_train), np.min(y_test), np.min(p_train), np.min(p_test)); hi = max(np.max(y_train), np.max(y_test), np.max(p_train), np.max(p_test))
    ax.scatter(y_train, p_train, s=14, alpha=.5, label=f"train  R² {met.r2_score(y_train, p_train):.3f}")
    ax.scatter(y_test, p_test, s=22, alpha=.85, label=f"test   R² {met.r2_score(y_test, p_test):.3f}")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel(f"actual {unit}"); ax.set_ylabel(f"predicted {unit}"); ax.set_title(title); ax.legend(); ax.set_aspect("equal", "box")
    return ax


def hyper_curve(model, param, values, X_train, y_train, X_test, y_test, ax=None, log=False):
    """Train and test score while one hyper-parameter changes (the loops of the course notebooks)."""
    tr, te = [], []
    for v in values:
        m = clone(model).set_params(**{param: v}).fit(X_train, y_train)
        tr.append(m.score(X_train, y_train)); te.append(m.score(X_test, y_test))
    ax = ax or plt.gca()
    ax.plot(values, tr, "o-", label="train"); ax.plot(values, te, "s-", label="test")
    ax.set_xlabel(param); ax.set_ylabel("score"); ax.legend()
    if log: ax.set_xscale("log")
    best = values[int(np.argmax(te))]
    ax.axvline(best, color="grey", ls=":"); ax.set_title(f"best {param} on this split = {best}")
    return pd.DataFrame({param: values, "train": tr, "test": te})


def compare(models, X_train, y_train, X_test, y_test, task="regression"):
    """Fit every model and return a table sorted by the test score."""
    rows = []
    for name, m in models.items():
        t0 = time.time(); m = clone(m).fit(X_train, y_train); sec = time.time() - t0
        if task == "regression":
            r = {"model": name, "R2_train": m.score(X_train, y_train), **{k + "_test": v for k, v in regression_metrics(y_test, m.predict(X_test)).items()}}
        else:
            pr = m.predict_proba(X_test) if hasattr(m, "predict_proba") else None
            r = {"model": name, "acc_train": m.score(X_train, y_train), **{k + "_test": v for k, v in classification_metrics(y_test, m.predict(X_test), pr, getattr(m, "classes_", None)).items()}}
        r["fit_s"] = sec; rows.append(r)
    key = "R2_test" if task == "regression" else "F1_macro_test"
    return pd.DataFrame(rows).sort_values(key, ascending=False).round(4).reset_index(drop=True)


def confusion(y, p, labels, ax=None, title="Confusion matrix", normalize=False):
    cm = met.confusion_matrix(y, p, labels=labels, normalize="true" if normalize else None)
    ax = ax or plt.gca()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right"); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title); ax.grid(False)
    return cm


def importance(model, X, y, n_repeats=20, seed=0, ax=None):
    """Permutation importance on held-out data: how much the score drops when one input is shuffled."""
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed)
    s = pd.Series(r.importances_mean, index=X.columns).sort_values()
    ax = ax or plt.gca(); s.plot.barh(ax=ax, xerr=pd.Series(r.importances_std, index=X.columns)[s.index]); ax.set_xlabel("drop in score when shuffled")
    return s.sort_values(ascending=False)
