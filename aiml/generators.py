"""Synthetic data generators for Part I: make as much data as you want, with known physics and controlled noise.

make_asphaltene_data  precipitation curves W(T, n-alkane, dilution) from an empirical model fitted to the course data
make_vfm_data         well tests from inflow (IPR) + tubing + Gilbert choke equations
make_eor_data         field projects sampled class by class; labels from data, or from Taber screening rules
taber_screen          the expert-rule baseline (Taber, Martin & Seright, 1997) to compare with machine learning

Command line:  python -m aiml.generators asphaltene|vfm|eor --n 500 --noise 0.03 --out my_data.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import brentq

# --------------------------------------------------------------------------------------------
# Asphaltene precipitation
# --------------------------------------------------------------------------------------------
ASPH_FIT = dict(a0=7.31, a1=0.371, a2=0.0651, dr0=1.36, b=7.43)   # fitted to the 150 clean tests (R2 0.95)


def asphaltene_model(T, C, DR, a0=7.31, a1=0.371, a2=0.0651, dr0=1.36, b=7.43):
    """wt% precipitated = plateau(T, C) * (1 - exp(-(DR - onset) / b)).
    Heavier n-alkanes (larger C) and higher temperature precipitate less."""
    A = np.clip(a0 - a1 * (np.asarray(C) - 5) - a2 * (np.asarray(T) - 293), 0.1, None)
    return A * (1 - np.exp(-np.clip(np.asarray(DR) - dr0, 0, None) / b))


def make_asphaltene_data(temperatures=(293, 308, 323, 332), carbon_numbers=(5, 6, 7, 8, 9, 10, 12), dilution_ratios=(3.6, 5.5, 9.7, 14.3, 20.1, 24.4),
                         noise=0.05, outliers=0.0, outlier_factor=(3, 8), seed=0, out=None):
    """One row per (temperature, n-alkane, dilution ratio). noise = relative measurement error;
    outliers = share of rows multiplied by a factor drawn from outlier_factor (to practise outlier detection)."""
    r = np.random.default_rng(seed)
    g = np.array(np.meshgrid(temperatures, carbon_numbers, dilution_ratios, indexing="ij")).reshape(3, -1).T
    T, C, DR = g[:, 0], g[:, 1], g[:, 2] * (1 + r.normal(0, 0.01, len(g)))
    w = asphaltene_model(T, C, DR) * (1 + r.normal(0, noise, len(g)))
    bad = r.random(len(g)) < outliers
    w[bad] *= r.uniform(*outlier_factor, bad.sum())
    df = pd.DataFrame({"temperature_K": T.astype(int), "alkane_carbon_number": C.astype(int), "dilution_ratio": DR.round(4),
                       "asphaltene_wt_pct": np.clip(w, 0, None).round(4), "is_outlier": bad})
    if out: df.to_csv(out, index=False)
    return df


# --------------------------------------------------------------------------------------------
# Virtual flow metering
# --------------------------------------------------------------------------------------------
def _well(choke_in, p_res, pi, wc, glr, depth_ft, psep, t_res=190.0, t_surf=60.0):
    """Solve one well: inflow q = PI (Pr - Pwf); tubing Pwf = Pwh + gradient * depth + friction;
    choke: Gilbert Pwh = 435 GLR^0.546 q / S^1.89 (S in 64ths), but never below the line pressure."""
    S = choke_in * 64.0
    grad = 0.433 * (0.85 + 0.15 * wc) * (0.45 + 0.35 / (1 + glr))           # psi/ft, lighter with more gas
    def resid(q):
        pwh = max(435.0 * glr ** 0.546 * q / S ** 1.89, psep + 150.0 + 2e-5 * q ** 2)
        pwf = pwh + grad * depth_ft + 2.5e-6 * q ** 2 * depth_ft / 1e3
        return pi * (p_res - pwf) - q
    q = brentq(resid, 1.0, pi * p_res)
    pwh = max(435.0 * glr ** 0.546 * q / S ** 1.89, psep + 150.0 + 2e-5 * q ** 2)
    pwf = p_res - q / pi
    wht = t_surf + (t_res - t_surf) * (1 - np.exp(-q / 2500.0))              # high rate = less heat lost on the way up
    return q, pwf, pwh, wht


def make_vfm_data(n_tests=600, chokes=(0.25, 0.32, 0.42, 0.54, 0.70, 0.90, 1.16, 1.50, 1.94, 2.50), p_res=(5300, 5700),
                  pi=(3.5, 6.5), water_cut=(0.0, 0.8), gor_scf_stb=(2500, 5500), depth_ft=8000, psep=(100, 600),
                  noise=0.01, water_cut_levels=None, seed=0, out=None):
    """Random well tests. Inputs are what a field measures (choke, BHP, WHP, WHT, Psep, Tsep); targets are the
    separator rates. noise = relative gauge error on pressures/temperatures (rates are exact).
    water_cut_levels: optional list such as (0, 0.2, 0.4, 0.6, 0.8) to draw water cut from fixed values, like the course data."""
    r = np.random.default_rng(seed); rows = []
    for i in range(n_tests):
        c = r.choice(chokes); pr = r.uniform(*p_res); p = r.uniform(*pi); wc = r.choice(water_cut_levels) if water_cut_levels is not None else r.uniform(*water_cut); gor = r.uniform(*gor_scf_stb); ps = r.choice(np.arange(psep[0], psep[1] + 1, 100))
        glr = gor * (1 - wc) / 1000.0                                           # Mscf per bbl of liquid
        q, pwf, pwh, wht = _well(c, pr, p, wc, max(glr, 0.05), depth_ft, ps)
        tsep = 60 + 0.45 * (wht - 60) * (1 - np.exp(-q / 3000))
        e = lambda v: v * (1 + r.normal(0, noise))
        rows.append({"test_id": i, "choke_in": c, "bhp_psi": round(e(pwf), 2), "whp_psi": round(e(pwh), 2), "wht_F": round(e(wht), 2), "psep_psi": int(ps),
                     "tsep_F": round(e(tsep), 2), "oil_rate_stb_d": round(q * (1 - wc), 1), "water_rate_bbl_d": round(q * wc, 1),
                     "gas_rate_mmscf_d": round(q * (1 - wc) * gor / 1e6, 3), "water_cut": round(wc, 4),
                     "reservoir_pressure_psi": round(pr, 1), "productivity_index": round(p, 3)})
    df = pd.DataFrame(rows)
    if out: df.to_csv(out, index=False)
    return df


# --------------------------------------------------------------------------------------------
# EOR screening
# --------------------------------------------------------------------------------------------
TABER = {   # simplified Taber, Martin & Seright (1997) screening criteria
    "Nitrogen":        dict(api=(35, 99), visc=(0, 0.4), depth=(6000, 99999), sat=(40, 100)),
    "Hydrocarbon":     dict(api=(23, 99), visc=(0, 3), depth=(4000, 99999), sat=(30, 100)),
    "CO2":             dict(api=(22, 99), visc=(0, 10), depth=(2500, 99999), sat=(20, 100)),
    "Polymer":         dict(api=(15, 99), visc=(10, 150), depth=(0, 9000), perm=(10, 1e9), temp=(0, 200), sat=(50, 100)),
    "Combustion":      dict(api=(10, 27), visc=(0, 5000), depth=(0, 11500), perm=(50, 1e9), sat=(50, 100)),
    "Steam Injection": dict(api=(8, 25), visc=(20, 2e5), depth=(0, 4500), perm=(200, 1e9), sat=(40, 100)),
}
_KEY = {"api": "api_gravity", "visc": "viscosity_cp", "depth": "depth_ft", "perm": "permeability_md", "temp": "temperature_F", "sat": "oil_saturation_pct"}


def taber_scores(df):
    """Share of the screening criteria each method meets, for every row (0-1)."""
    out = {}
    for m, crit in TABER.items():
        ok = np.mean([(df[_KEY[k]] >= lo) & (df[_KEY[k]] <= hi) for k, (lo, hi) in crit.items()], axis=0)
        out[m] = ok
    return pd.DataFrame(out, index=df.index)


TABER_ORDER = ["Polymer", "Steam Injection", "Nitrogen", "CO2", "Hydrocarbon", "Combustion"]   # narrowest windows first


def taber_screen(df):
    """Expert-system baseline: the first method (in TABER_ORDER) whose criteria are all met;
    if none passes, the method that meets the largest share of its criteria."""
    S = taber_scores(df)[TABER_ORDER]; full = S >= 0.999
    return pd.Series(np.where(full.any(axis=1), full.idxmax(axis=1), S.idxmax(axis=1)), index=df.index)


def make_eor_data(n=600, source=None, label="data", noise=0.15, seed=0, out=None):
    """Sample synthetic projects class by class from the log-normal spread of the real course data.
    label = "data": keep the class they were sampled from; "taber": relabel with the screening rules."""
    from .data import load_eor, EOR_FEATURES
    r = np.random.default_rng(seed)
    src = source if source is not None else load_eor()
    share = src.eor_method.value_counts(normalize=True)
    rows = []
    for m, p in share.items():
        g = src[src.eor_method == m][EOR_FEATURES].clip(lower=0.1)
        mu, sd = np.log(g).mean(), np.log(g).std().fillna(0.3) + noise
        k = max(int(round(n * p)), 3)
        X = np.exp(r.normal(mu.values, sd.values, (k, len(EOR_FEATURES))))
        d = pd.DataFrame(X, columns=EOR_FEATURES); d["eor_method"] = m; rows.append(d)
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["oil_saturation_pct"] = df.oil_saturation_pct.clip(5, 98)
    df = df.round({"permeability_md": 1, "depth_ft": 0, "api_gravity": 1, "viscosity_cp": 2, "temperature_F": 1, "oil_saturation_pct": 1})
    if label == "taber":
        df["eor_method"] = taber_screen(df)
    if out: df.to_csv(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Part I data generators")
    ap.add_argument("kind", choices=["asphaltene", "vfm", "eor"]); ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--noise", type=float, default=None); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = a.out or f"synthetic_{a.kind}.csv"
    if a.kind == "asphaltene": df = make_asphaltene_data(noise=0.05 if a.noise is None else a.noise, seed=a.seed, out=out)
    elif a.kind == "vfm": df = make_vfm_data(n_tests=a.n, noise=0.01 if a.noise is None else a.noise, seed=a.seed, out=out)
    else: df = make_eor_data(n=a.n, noise=0.15 if a.noise is None else a.noise, seed=a.seed, out=out)
    print(f"wrote {out}: {len(df)} rows"); print(df.head())


if __name__ == "__main__":
    main()
