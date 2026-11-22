"""Data preparation helpers (Module 04 of the slides).
detect_outliers_iqr / remove_outliers follow the course's outlier_detection.py."""
import numpy as np
import pandas as pd


def detect_outliers_iqr(data, column, threshold=1.5):
    """Return (lower_bound, upper_bound, outlier_index) with the interquartile-range rule."""
    q1, q3 = data[column].quantile(0.25), data[column].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
    return lo, hi, data[(data[column] < lo) | (data[column] > hi)].index


def remove_outliers(data, column, lower, upper):
    return data[(data[column] >= lower) & (data[column] <= upper)]


def outlier_table(df, columns=None, threshold=1.5):
    """IQR summary for every numeric column."""
    columns = columns or df.select_dtypes("number").columns
    rows = []
    for c in columns:
        lo, hi, idx = detect_outliers_iqr(df, c, threshold)
        rows.append({"column": c, "lower": round(lo, 3), "upper": round(hi, 3), "n_outliers": len(idx), "share_%": round(100 * len(idx) / len(df), 1)})
    return pd.DataFrame(rows)


def zscore_outliers(data, column, z=3.0):
    s = (data[column] - data[column].mean()) / data[column].std()
    return data.index[s.abs() > z]


def plot_outliers(data, column, ax=None):
    import matplotlib.pyplot as plt
    lo, hi, idx = detect_outliers_iqr(data, column)
    ax = ax or plt.gca()
    ax.scatter(data.index, data[column], s=12, alpha=.6, label="data")
    ax.scatter(idx, data.loc[idx, column], s=22, c="red", label=f"outliers ({len(idx)})")
    for b in (lo, hi):
        ax.axhline(b, color="r", ls="--", lw=1)
    ax.set_title(f"IQR outliers · {column}"); ax.set_xlabel("row"); ax.legend()
    return ax
