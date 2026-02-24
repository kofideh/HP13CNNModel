#!/usr/bin/env python3
# sim_export_enriched.py
# Adds estimator option: --estimate nn (HybridMultiHead)

import argparse, math, json, csv, os
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# ---------------- optional torch / model import (only needed if --estimate nn) ------------
_torch = None
_HybridMultiHead = None
def _lazy_import_nn():
    global _torch, _HybridMultiHead
    if _torch is None:
        import torch as _torch  # noqa: F401 (kept in module global)
    if _HybridMultiHead is None:
        try:
            from hybrid_model_utils import HybridMultiHead as _HybridMultiHead  # noqa: F401
        except Exception as e:
            raise ImportError(
                "Could not import HybridMultiHead from hybrid_model_utils. "
                "Ensure the module is on PYTHONPATH."
            ) from e

# ---------- helpers ----------
def set_seed(seed: int):
    rng = np.random.default_rng(seed); np.random.seed(seed); return rng

def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(y, x))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)

def partial_r2(y: np.ndarray, X_conf: np.ndarray, X_full: np.ndarray) -> float:
    beta_r, *_ = np.linalg.lstsq(X_conf, y, rcond=None)
    yhat_r = X_conf @ beta_r
    sse_r = np.sum((y - yhat_r) ** 2)
    beta_f, *_ = np.linalg.lstsq(X_full, y, rcond=None)
    yhat_f = X_full @ beta_f
    sse_f = np.sum((y - yhat_f) ** 2)
    if not np.isfinite(sse_r) or sse_r <= 0: return np.nan
    return float(np.clip((sse_r - sse_f) / sse_r, 0.0, 1.0))

def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    x_ = np.asarray(x, float); y_ = np.asarray(y, float)
    m = (x_ > 0) & (y_ > 0) & np.isfinite(x_) & np.isfinite(y_)
    if np.count_nonzero(m) < 2: return np.nan
    X = np.vstack([np.log(x_[m]), np.ones(np.count_nonzero(m))]).T
    b, a = np.linalg.lstsq(X, np.log(y_[m]), rcond=None)[0]
    return float(b)

def parse_float_list(s: str, default: List[float]) -> List[float]:
    if not s: return default
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_float_list2(s):
    return [float(x) for x in s.split(",") if str(x).strip() != ""]

def parse_shift_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",")] if s else [-10.0, -5.0, 0.0, 5.0, 10.0]

def cov(x):
    x = np.asarray(x, float); m = np.nanmean(x); s = np.nanstd(x, ddof=1)
    return float(s/m) if (m not in (0, np.nan) and np.isfinite(m)) else np.nan

def r2_linreg(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 3: return np.nan
    X = np.column_stack([np.ones(ok.sum()), x[ok]])
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    yhat = X @ beta
    return r2_score(y[ok], yhat)

def r2_corr(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 3: return np.nan
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    return float(r*r)

def robust_peak_1d(x: np.ndarray, q: float = 99.9) -> float:
    x = np.asarray(x, float)
    if x.size == 0: return 0.0
    return float(np.percentile(np.abs(x), q))

# ---------- acquisition ----------
def vfa_schedule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    th_p, th_l = [], []
    for i in range(n):
        val = math.atan(1.0 / math.sqrt(max(1, 16 - i))) if i < n-1 else math.radians(90.0)
        th_p.append(val); th_l.append(val)
    return np.array(th_p), np.array(th_l)

def cfa_schedule(n: int, pyr_deg: float = 11.0, lac_deg: float = 80.0) -> Tuple[np.ndarray, np.ndarray]:
    return np.full(n, math.radians(pyr_deg)), np.full(n, math.radians(lac_deg))

def r1_app(r1: float, theta: np.ndarray, TR: float) -> np.ndarray:
    return r1 - np.log(np.cos(np.clip(theta, 0, math.radians(89.9)))) / TR

# ---------- AIF ----------
def aif_gamma(t: np.ndarray, A: float, t0: float, alpha: float, beta: float) -> np.ndarray:
    tp = np.maximum(t - t0, 0.0); return A * (tp ** alpha) * np.exp(-tp / beta)

def aif_infusion(t: np.ndarray, A: float, t0: float, tau: float, ramp: bool = False) -> np.ndarray:
    tp = t - t0
    if ramp:
        x = np.clip(1 - np.abs((tp - tau/2) / (tau/2)), 0, 1); return A * x
    return A * ((tp >= 0) & (tp <= tau)).astype(float)

def normalize_area(aif_func, t: np.ndarray, target_area: float, **kwargs) -> Tuple[np.ndarray, float]:
    y = aif_func(t, **kwargs); area = trapz(y, t); scale = target_area / max(area, 1e-12)
    return y * scale, scale

# ---------- two-state ODE ----------
@dataclass
class KineticParams:
    kpl: float; kve: float; vb: float; r1p: float; r1l: float

def integrate_two_state(t, U, theta_p, theta_l, TR, pars: KineticParams):
    R1p_app = r1_app(pars.r1p, theta_p, TR)
    R1l_app = r1_app(pars.r1l, theta_l, TR)
    def rhs(_t, y):
        Ui = np.interp(_t, t, U)
        R1p_i = np.interp(_t, t, R1p_app)
        R1l_i = np.interp(_t, t, R1l_app)
        Pe, L = y
        dPe = (pars.kve/pars.vb)*Ui - ((pars.kve*(1-pars.vb)/pars.vb)+pars.kpl+R1p_i)*Pe
        dL  = pars.kpl*Pe - R1l_i*L
        return [dPe, dL]
    sol = solve_ivp(rhs, (t[0], t[-1]), y0=[0.0, 0.0], t_eval=t, rtol=1e-6, atol=1e-9)
    Pe = sol.y[0]; L = sol.y[1]; Sp = pars.vb*U + (1-pars.vb)*Pe
    return Pe, L, Sp

# ---------- estimators (NLLS variants) ----------
def fit_measured_driver_lactate(t, S_lac_obs, U_driver, theta_p, theta_l, TR, init, bounds, r1p, r1l):
    def model_lac(_t, kpl, kve, vb):
        pars = KineticParams(kpl=kpl, kve=kve, vb=vb, r1p=r1p, r1l=r1l)
        _, Lp, _ = integrate_two_state(t, U_driver, theta_p, theta_l, TR, pars)
        return Lp
    try:
        popt, _ = curve_fit(model_lac, t, S_lac_obs, p0=init, bounds=bounds, maxfev=3000)
        kpl_hat, kve_hat, vb_hat = popt
    except Exception:
        kpl_hat, kve_hat, vb_hat = np.nan, np.nan, np.nan
    return {"kpl_hat": float(kpl_hat), "kve_hat": float(kve_hat), "vb_hat": float(vb_hat)}

def fit_joint_pyruvate_lactate(t, S_pyr_obs, S_lac_obs, theta_p, theta_l, TR, init, bounds, r1p, r1l):
    S_pyr_smooth = savgol_filter(S_pyr_obs, window_length=max(5, (len(t)//5)*2+1), polyorder=2)
    def residuals(param_triplet):
        kpl, kve, vb = param_triplet
        pars = KineticParams(kpl=kpl, kve=kve, vb=vb, r1p=r1p, r1l=r1l)
        U0 = (S_pyr_smooth - (1 - vb) * 0.0) / max(vb, 1e-6)
        Pe, _, _ = integrate_two_state(t, U0, theta_p, theta_l, TR, pars)
        U1 = (S_pyr_smooth - (1 - vb) * Pe) / max(vb, 1e-6)
        U1 = np.clip(U1, 0, None)
        Pe2, L2, Sp2 = integrate_two_state(t, U1, theta_p, theta_l, TR, pars)
        return np.concatenate([Sp2 - S_pyr_obs, L2 - S_lac_obs])
    def concat_model(_t, kpl, kve, vb): return residuals((kpl, kve, vb))
    target = np.zeros(2 * len(t))
    try:
        popt, _ = curve_fit(concat_model, np.zeros_like(target), target,
                            p0=init, bounds=bounds, maxfev=5000)
        kpl_hat, kve_hat, vb_hat = popt
    except Exception:
        kpl_hat, kve_hat, vb_hat = np.nan, np.nan, np.nan
    return {"kpl_hat": float(kpl_hat), "kve_hat": float(kve_hat), "vb_hat": float(vb_hat)}

# ---------- sim core ----------
@dataclass
class SimConfig:
    TR: float = 2.0
    n_timepoints: int = 30
    schedule: str = "VFA"      # "VFA" or "CFA"
    cfa_pyr_deg: float = 11.0
    cfa_lac_deg: float = 80.0
    snr_pyr: float = 32.0
    snr_lac: float = 20.0
    r1p: float = 1/30.0
    r1l: float = 1/25.0

import numpy as np, math
from numpy import trapz

def _ratio_metric(S_l, S_p, t, ratio_mode="auc_full",
                  ratio_tmin=0.0, ratio_tmax=None, ratio_tpoint=None):
    """
    Returns (auc_l, auc_p, auc_ratio) where auc_ratio is computed according to ratio_mode.
    S_l and S_p are already noisy signals.
    """
    t = np.asarray(t)
    S_l = np.asarray(S_l)
    S_p = np.asarray(S_p)

    # clip negatives for AUC stability (consistent with your current code)
    Sl = np.maximum(S_l, 0.0)
    Sp = np.maximum(S_p, 0.0)

    if ratio_mode == "auc_full":
        auc_p = trapz(Sp, t)
        auc_l = trapz(Sl, t)
        auc_ratio = (auc_l / auc_p) if auc_p > 0 else np.nan
        return float(auc_l), float(auc_p), float(auc_ratio)

    if ratio_mode == "auc_window":
        if ratio_tmax is None:
            ratio_tmax = float(t[-1])
        mask = (t >= ratio_tmin) & (t <= ratio_tmax)
        if np.count_nonzero(mask) < 2:
            return np.nan, np.nan, np.nan
        auc_p = trapz(Sp[mask], t[mask])
        auc_l = trapz(Sl[mask], t[mask])
        auc_ratio = (auc_l / auc_p) if auc_p > 0 else np.nan
        return float(auc_l), float(auc_p), float(auc_ratio)

    if ratio_mode == "tpoint":
        if ratio_tpoint is None:
            ratio_tpoint = float(t[len(t)//2])
        # nearest neighbor index
        idx = int(np.argmin(np.abs(t - ratio_tpoint)))
        denom = Sp[idx]
        auc_ratio = (Sl[idx] / denom) if denom > 0 else np.nan
        # auc_p/auc_l returned as NaN since this is not an AUC
        return np.nan, np.nan, float(auc_ratio)

    raise ValueError(f"Unknown ratio_mode={ratio_mode}")


def simulate_one(cfg: SimConfig, pars_true: KineticParams,
                 aif_mode="bolus", t_shift=0.0, gamma_gain=1.0,
                 delta_theta_p_deg=0.0, delta_theta_l_deg=0.0,
                 # NEW: AIF-shape controls
                 aif_alpha=2.0, aif_beta=6.0, aif_t0=2.0, aif_tau=20.0,
                 rng=None):
    n = cfg.n_timepoints
    t = np.arange(n) * cfg.TR

    if cfg.schedule.upper() == "VFA":
        th_p, th_l = vfa_schedule(n)
    else:
        th_p, th_l = cfa_schedule(n, cfg.cfa_pyr_deg, cfg.cfa_lac_deg)

    th_p = np.clip(th_p + math.radians(delta_theta_p_deg), 0, math.radians(89.9))
    th_l = np.clip(th_l + math.radians(delta_theta_l_deg), 0, math.radians(89.9))

    tf = np.linspace(0, t[-1], 2001)

    # --- Reference dose (area) for normalization (keeps total delivery constant) ---
    ref = aif_gamma(tf, A=1.0, t0=2.0, alpha=2.0, beta=6.0)
    dose_ref = trapz(ref, tf)

    # --- Build AIF at high-res with requested shape, then normalize to dose_ref ---
    if aif_mode.lower() == "bolus":
        # Gamma-variate bolus with variable (alpha,beta,t0), normalized to fixed area
        U_hr, _ = normalize_area(
            lambda tt, **kw: aif_gamma(tt, **kw),
            tf, dose_ref,
            A=1.0, t0=float(aif_t0), alpha=float(aif_alpha), beta=float(aif_beta)
        )

    elif aif_mode.lower() == "infusion":
        # Infusion with variable (t0,tau), normalized to fixed area
        U_hr, _ = normalize_area(
            lambda tt, **kw: aif_infusion(tt, **kw),
            tf, dose_ref,
            A=1.0, t0=float(aif_t0), tau=float(aif_tau), ramp=False
        )
    else:
        raise ValueError("aif_mode must be 'bolus' or 'infusion' (or handled upstream for 'both').")

    # --- Apply acquisition-window shift (arrival variability) ---
    # (your original code double-interps; this is equivalent but cleaner)
    U_shifted = np.interp(tf - float(t_shift), tf, U_hr, left=0.0, right=0.0)

    # sample at acquisition times and apply pyruvate-only gain
    U = np.interp(t, tf, U_shifted) * float(gamma_gain)

    # forward model
    Pe, L, S_p = integrate_two_state(t, U, th_p, th_l, cfg.TR, pars_true)
    S_l = (1 - pars_true.vb) * L

    rng = rng or np.random.default_rng()

    def add_noise(x, snr):
        sigma = max(np.max(np.abs(x)), 1e-8) / max(snr, 1e-3)
        return x + rng.normal(0, sigma, size=x.shape)

    S_p_noisy = add_noise(S_p, cfg.snr_pyr)
    S_l_noisy = add_noise(S_l, cfg.snr_lac)

    auc_p = trapz(np.maximum(S_p_noisy, 0), t)
    auc_l = trapz(np.maximum(S_l_noisy, 0), t)
    auc_ratio = (auc_l / auc_p) if auc_p > 0 else np.nan

    return {
        "t": t, "theta_p": th_p, "theta_l": th_l,
        "S_p": S_p_noisy, "S_l": S_l_noisy,
        "U_used": U, "auc_ratio": float(auc_ratio),
        # NEW: keep the actual AIF parameters used (for CSV)
        "aif_alpha": float(aif_alpha), "aif_beta": float(aif_beta),
        "aif_t0": float(aif_t0), "aif_tau": float(aif_tau),
    }



import pandas as pd

def estimate_kpl_hat(sim, cfg, args, nn_ctx, init_kpl, init_vb, r1p, r1l):
    """
    sim: dict from simulate_one() containing t, S_p, S_l, U_used, theta_p, theta_l
    Returns: float kpl_hat (NaN if estimate=='none')
    """
    if args.estimate == "none":
        return float("nan")

    # keep your bounds/init exactly as you used elsewhere
    bounds = ([0.0, 0.0, 0.005], [0.5, 1.0, 0.20])
    init = (float(init_kpl), float(args.kve), float(init_vb))

    if args.estimate == "measured_driver":
        res = fit_measured_driver_lactate(
            sim["t"], sim["S_l"], sim["U_used"],
            sim["theta_p"], sim["theta_l"], cfg.TR,
            init, bounds, r1p, r1l
        )
        return float(res["kpl_hat"])

    if args.estimate == "joint":
        res = fit_joint_pyruvate_lactate(
            sim["t"], sim["S_p"], sim["S_l"],
            sim["theta_p"], sim["theta_l"], cfg.TR,
            init, bounds, r1p, r1l
        )
        return float(res["kpl_hat"])

    if args.estimate == "nn":
        # your existing nn_ctx wrapper
        return float(nn_ctx["predict_kpl"](sim["S_p"], sim["S_l"]))

    raise ValueError(f"Unknown --estimate {args.estimate}")


def run_matchratio(cfg, args, estimate_fn=None):
    """
    cfg: SimConfig
    args: argparse namespace (needs roc_kpl_neg/pos, vb_list, t_shift_list, gamma_list, etc.)
    estimate_fn: optional function(df)->df that adds kpl_hat (and maybe kve_hat/vb_hat)
    """
    rng = np.random.default_rng(args.seed)
    resample = set([s.strip() for s in args.match_resample.split(",") if s.strip()])

    vb_list = [float(x) for x in args.vb_list.split(",")] if hasattr(args, "vb_list") else [0.02]
    tsh_list = [float(x) for x in args.t_shift_list.split(",")] if hasattr(args, "t_shift_list") else [-10,-5,0,5,10]
    gamma_list = [float(x) for x in args.gamma_list.split(",")] if hasattr(args, "gamma_list") else [0.5,0.75,1.0,1.25,1.5]

    def sample_confounds():
        vb = float(rng.choice(vb_list)) if ("vb" in resample) else float(getattr(args, "vb", 0.02))
        tsh = float(rng.choice(tsh_list)) if ("tshift" in resample) else float(getattr(args, "t_shift", 0.0))
        gam = float(rng.choice(gamma_list)) if ("gamma" in resample) else 1.0
        return vb, tsh, gam

    rows = []
    for match_id in range(args.n_per_cell):
        # NEG sample
        vb0, tsh0, gam0 = sample_confounds()
        pars_neg = KineticParams(kpl=args.roc_kpl_neg, kve=args.kve, vb=vb0,
                                 r1p=cfg.r1p, r1l=cfg.r1l)  # adapt if your KineticParams differs
        sim_neg = simulate_one(
            cfg, pars_neg,
            aif_mode=args.aif_mode, t_shift=tsh0, gamma_gain=gam0,
            ratio_mode=args.ratio_mode, ratio_tmin=args.ratio_tmin,
            ratio_tmax=args.ratio_tmax, ratio_tpoint=args.ratio_tpoint,
            rng=rng
        )
        target = sim_neg["auc_ratio"]

        # POS sample: resample confounds until auc_ratio matches NEG
        sim_pos = None
        vb1=tsh1=gam1=None
        for _ in range(args.match_max_tries):
            vb1, tsh1, gam1 = sample_confounds()
            pars_pos = KineticParams(kpl=args.roc_kpl_pos, kve=args.kve, vb=vb1,
                                     r1p=cfg.r1p, r1l=cfg.r1l)
            cand = simulate_one(
                cfg, pars_pos,
                aif_mode=args.aif_mode, t_shift=tsh1, gamma_gain=gam1,
                ratio_mode=args.ratio_mode, ratio_tmin=args.ratio_tmin,
                ratio_tmax=args.ratio_tmax, ratio_tpoint=args.ratio_tpoint,
                rng=rng
            )
            if np.isfinite(target) and np.isfinite(cand["auc_ratio"]) and abs(cand["auc_ratio"] - target) <= args.match_tol:
                sim_pos = cand
                break

        if sim_pos is None:
            continue

        # Pack scalars only (avoid arrays in CSV unless you explicitly want them)
        rows.append({
            "y": 0, "match_id": match_id,
            "kpl_true": args.roc_kpl_neg, "kve_true": args.kve, "vb_true": vb0,
            "t_shift": tsh0, "gamma": gam0, "auc_ratio": sim_neg["auc_ratio"],
            "auc_p": sim_neg["auc_p"], "auc_l": sim_neg["auc_l"]
        })
        rows.append({
            "y": 1, "match_id": match_id,
            "kpl_true": args.roc_kpl_pos, "kve_true": args.kve, "vb_true": vb1,
            "t_shift": tsh1, "gamma": gam1, "auc_ratio": sim_pos["auc_ratio"],
            "auc_p": sim_pos["auc_p"], "auc_l": sim_pos["auc_l"]
        })

    df = pd.DataFrame(rows)

    # Apply estimator if provided (adds kpl_hat)
    if estimate_fn is not None and len(df) > 0:
        df = estimate_fn(df)

    return df


def safe_r2(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.nansum((y_true - np.nanmean(y_true))**2)
    if denom < eps:
        return np.nan
    return float(1.0 - (np.nansum((y_true - y_pred)**2) / denom))

def safe_std(x):
    x = np.asarray(x, float)
    n = np.sum(np.isfinite(x))
    return float(np.nanstd(x, ddof=1)) if n >= 2 else float("nan")


# ---------- stats csv ----------
# def write_stats_csv(stats_out, rows: List[Dict], experiment: str):
#     if not stats_out: return
#     if experiment == "gain": group_keys = ["gamma"]
#     elif experiment == "vb": group_keys = ["vb_true"]
#     elif experiment == "kpl": group_keys = ["kpl_true"]
#     elif experiment == "t1flip": group_keys = ["r1p_true","r1l_true","flip_delta_p","flip_delta_l"]
#     elif experiment == "tshift": group_keys = ["t_shift"]
#     elif experiment == "roc": group_keys = ["roc_group"] if "roc_group" in rows[0] else []
#     elif experiment == "trsweep": group_keys = ["TR"]
#     elif experiment == "aifshape":
#         group_keys = ["aif_mode","aif_alpha","aif_beta","aif_t0","aif_tau","t_shift"]
#     elif experiment == "matchratio":
#         group_keys = ["aif_mode"]
#     else: group_keys = []

#     from collections import defaultdict
#     buckets = defaultdict(list)
#     for r in rows:
#         key = tuple(r[k] for k in group_keys) if group_keys else ("all",)
#         buckets[key].append(r)

#     out_rows = []
#     for key, rr in buckets.items():
#         lp = np.array([x["auc_ratio"] for x in rr], float)
#         kp = np.array([x["kpl_hat"] for x in rr], float)
#         row = {}
#         for k, v in zip(group_keys, key if group_keys else ()): row[k] = v
#         row.update({
#             "n": int(len(rr)),
#             "mean_LacPyr": float(np.nanmean(lp)), "std_LacPyr": float(np.nanstd(lp, ddof=1)), "cov_LacPyr": cov(lp),
#             "mean_kPLhat": float(np.nanmean(kp)), "std_kPLhat": float(np.nanstd(kp, ddof=1)), "cov_kPLhat": cov(kp),
#         })
#         if experiment == "roc":
#             y = np.array([x["is_pos"] for x in rr], int)
#             auc_lp = auc_from_scores(y, lp)
#             auc_kp = auc_from_scores(y, kp)
#             row["auc_LacPyr"] = auc_lp
#             row["auc_kPLhat"] = auc_kp
#         out_rows.append(row)

#     with open(stats_out, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
#         writer.writeheader(); writer.writerows(out_rows)
import csv
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from dataclasses import is_dataclass, replace

def cfg_with(cfg, **kwargs):
    # works for dataclass SimConfig or simple class with __dict__
    if is_dataclass(cfg):
        return replace(cfg, **kwargs)
    d = dict(cfg.__dict__)
    d.update(kwargs)
    return cfg.__class__(**d)


def _cov(x: np.ndarray, eps: float = 1e-8) -> float:
    """Coefficient of variation with guard against tiny means."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    denom = max(abs(mu), eps)
    return sd / denom

def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m): return np.nan
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))

def _weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted SD (population-style) across strata; OK for summarizing across cells."""
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if np.sum(m) < 2: return np.nan
    mu = np.sum(w[m] * x[m]) / np.sum(w[m])
    var = np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m])
    return float(np.sqrt(var))

def _experiment_keys(experiment: str) -> Tuple[List[str], List[str]]:
    """
    Returns (confound_keys, truth_keys) for TWO-LEVEL robustness:
      - confound_keys: what you’re sweeping / want on x-axis
      - truth_keys: what should be held fixed for robustness (kpl_true, vb_true, etc.)
    """
    if experiment == "gain":
        return (["gamma"], ["kpl_true", "vb_true"])
    if experiment == "vb":
        return (["vb_true"], ["kpl_true"])
    if experiment == "tshift":
        return (["t_shift"], ["kpl_true", "vb_true"])
    if experiment == "t1flip":
        return (["r1p_true", "r1l_true", "flip_delta_p", "flip_delta_l"], ["kpl_true", "vb_true"])
    if experiment == "aifshape":
        # include t_shift if you sweep it during aifshape
        return (["aif_mode", "aif_alpha", "aif_beta", "aif_t0", "aif_tau", "t_shift"], ["kpl_true", "vb_true"])
    if experiment == "kpl":
        # This is not a “robustness-to-confound” sweep; treat vb as truth stratum
        return (["kpl_true"], ["vb_true"])
    if experiment == "matchratio":
        # If you want robustness summaries per AIF mode (not paired deltas)
        return (["aif_mode"], [])  # note: matchratio is better analyzed with paired deltas, not CoV
    if experiment == "trsweep":
        return (["TR"], ["kpl_true","vb_true"])

    # fallback
    return ([], [])

def write_stats_csv_twolevel(stats_out: str, rows: List[Dict], experiment: str,
                            min_rep: int = 5, eps_cov: float = 1e-8,
                            write_cell_csv: bool = True) -> None:
    """
    Two-level robustness stats:
      Level 1: within each (confound_keys + truth_keys) cell -> CoV over noise replicates
      Level 2: aggregate across truth strata within each confound bucket -> mean/SD CoV
    Writes:
      - stats_out: aggregated (confound-level) CSV
      - optional: stats_out with suffix "_cell.csv": cell-level CSV
    """
    if not stats_out:
        return
    if not rows:
        raise ValueError("write_stats_csv_twolevel: rows is empty")

    conf_keys, truth_keys = _experiment_keys(experiment)
    df = pd.DataFrame(rows)

    # Required columns
    for col in ["auc_ratio", "kpl_hat"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in rows")

    group1 = conf_keys + truth_keys
    if not group1:
        # no grouping defined -> treat as one bucket
        group1 = ["__all__"]
        df["__all__"] = "all"

    # ----- Level 1: compute cell-level CoV -----
    cell_rows = []
    for keys, g in df.groupby(group1, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        lp = g["auc_ratio"].to_numpy(float)
        kp = g["kpl_hat"].to_numpy(float)

        # effective sample counts (finite only)
        n_lp = int(np.isfinite(lp).sum())
        n_kp = int(np.isfinite(kp).sum())
        n_eff = int(min(n_lp, n_kp))

        if n_eff < min_rep:
            continue

        row = {k: v for k, v in zip(group1, keys)}
        row.update({
            "n_eff": n_eff,
            "mean_LacPyr": float(np.nanmean(lp)),
            "std_LacPyr": float(np.nanstd(lp, ddof=1)),
            "cov_LacPyr": float(_cov(lp, eps=eps_cov)),
            "mean_kPLhat": float(np.nanmean(kp)),
            "std_kPLhat": float(np.nanstd(kp, ddof=1)),
            "cov_kPLhat": float(_cov(kp, eps=eps_cov)),
        })
        cell_rows.append(row)

    cell_df = pd.DataFrame(cell_rows)
    if cell_df.empty:
        raise ValueError(
            f"No cells met min_rep={min_rep}. "
            "Increase n_per_cell or lower min_rep."
        )

    # Write cell-level CSV (optional)
    if write_cell_csv:
        base, ext = os.path.splitext(stats_out)
        cell_path = f"{base}_cell{ext if ext else '.csv'}"
        cell_df.to_csv(cell_path, index=False)

    # ----- Level 2: aggregate across truth strata within each confound bucket -----
    if conf_keys:
        group2 = conf_keys
    else:
        group2 = ["__all__"]
        cell_df["__all__"] = "all"

    agg_rows = []
    for keys, g in cell_df.groupby(group2, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        w = g["n_eff"].to_numpy(float)

        cov_lp = g["cov_LacPyr"].to_numpy(float)
        cov_kp = g["cov_kPLhat"].to_numpy(float)

        out = {k: v for k, v in zip(group2, keys)}
        out.update({
            "n_cells": int(len(g)),
            "n_eff_total": int(np.sum(w)),
            # weighted across truth strata
            "mean_cov_LacPyr": _weighted_mean(cov_lp, w),
            "std_cov_LacPyr": _weighted_std(cov_lp, w),
            "mean_cov_kPLhat": _weighted_mean(cov_kp, w),
            "std_cov_kPLhat": _weighted_std(cov_kp, w),
            # optional: also report average means (helps interpret sign/scale)
            "mean_mean_LacPyr": _weighted_mean(g["mean_LacPyr"].to_numpy(float), w),
            "mean_mean_kPLhat": _weighted_mean(g["mean_kPLhat"].to_numpy(float), w),
        })
        agg_rows.append(out)

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(stats_out, index=False)

# ---------- AUROC helpers ----------
def auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, float)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m]; s = s[m]
    if (y==1).sum()==0 or (y==0).sum()==0: return np.nan
    # rank-based AUC (Mann–Whitney)
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]; y_sorted = y[order]
    ranks = np.empty_like(s_sorted, dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j+1 < len(s_sorted) and s_sorted[j+1] == s_sorted[i]:
            j += 1
        ranks[i:j+1] = np.mean(np.arange(i+1, j+2))
        i = j + 1
    r_pos = ranks[y_sorted==1]
    n_pos = float((y==1).sum()); n_neg = float((y==0).sum())
    auc = (np.sum(r_pos) - n_pos*(n_pos+1)/2.0) / (n_pos*n_neg)
    return float(auc)

# ---------- main experiment ----------
def run_experiment(args) -> None:
    rng = set_seed(args.seed)
    cfg = SimConfig(
        TR=args.TR, n_timepoints=args.n_timepoints, schedule=args.schedule,
        cfa_pyr_deg=args.cfa_pyr_deg, cfa_lac_deg=args.cfa_lac_deg,
        snr_pyr=args.snr_pyr, snr_lac=args.snr_lac, r1p=args.r1p, r1l=args.r1l
    )

    # Optional: load NN once
    nn_ctx = None
    if args.estimate == "nn":
        _lazy_import_nn()
        if not os.path.exists(args.nn_weights):
            raise FileNotFoundError(f"--nn_weights not found: {args.nn_weights}")
        input_dim_raw = cfg.n_timepoints * 2
        input_dim_norm = cfg.n_timepoints * 2
        model = _HybridMultiHead(input_dim_raw=input_dim_raw, input_dim_norm=input_dim_norm)
        state = _torch.load(args.nn_weights, map_location=_torch.device("cpu"))
        model.load_state_dict(state)
        model.eval()
        # build predictor closure
        def nn_predict_kpl(S_p: np.ndarray, S_l: np.ndarray) -> float:
            # robust-peak normalization (per-sample) as in training
            comb = np.concatenate([S_p, S_l], axis=0)
            p_exam = robust_peak_1d(comb, q=99.9)
            alpha = (args.nn_ptrain / max(p_exam, 1e-12)) if args.nn_use_robust_peak else 1.0
            Sp_n = S_p * alpha
            Sl_n = S_l * alpha
            Xn = np.concatenate([Sp_n, Sl_n]).astype(np.float32)[None, :]
            Xr = np.concatenate([S_p, S_l]).astype(np.float32)[None, :]
            with _torch.no_grad():
                Xn_t = _torch.from_numpy(Xn)
                Xr_t = _torch.from_numpy(Xr)
                y = model(Xn_t, Xr_t).cpu().numpy()
            return float(y[0, args.nn_kpl_index])
        nn_ctx = {"predict_kpl": nn_predict_kpl}

    # defaults/grids
    gamma_grid = np.array([0.5,0.75,1.0,1.25,1.5]) if args.experiment == "gain" else np.array([1.0])
    vb_grid    = np.array([0.01,0.02,0.04,0.06,0.08]) if args.experiment == "vb" else np.array([0.02])
    kpl_grid   = np.array([0.12,0.132,0.144]) if args.experiment == "kpl" else np.array([0.132])
    t1p_list = parse_float_list(args.t1p_list, [30.0])
    t1l_list = parse_float_list(args.t1l_list, [25.0])
    dthp_list = parse_float_list(args.flip_delta_p_list, [0.0])
    dthl_list = parse_float_list(args.flip_delta_l_list, [0.0])
    t_shift_grid = np.array([args.t_shift])
    t_shift_list = np.array(parse_shift_list(args.t_shift_list), float)
    gamma_list_roc = np.array(parse_float_list(args.gamma_list, [0.5,0.75,1.0,1.25,1.5]), float)
    vb_list_roc    = np.array(parse_float_list(args.vb_list, [0.01,0.02,0.04,0.06,0.08]), float)

    aif_modes = ["bolus","infusion"] if args.aif_mode == "both" else [args.aif_mode]
    rows: List[Dict] = []

    for aif_mode in aif_modes:
        # ---- TR sweep with constant window ----
        if args.experiment == "trsweep":
            tr_list = parse_float_list2(args.tr_list)  # you already have parse_float_list2(...)
            t_end = float(args.t_end)

            # Confounds to (optionally) resample and then HOLD FIXED across TR for each match_id
            rng = np.random.default_rng(args.seed)

            # Choose how much confounding you want *in* this invariance test:
            # (A) Pure protocol-only: fix confounds
            def sample_confounds(_rng):
                return float(1.0), float(0.0)  # (gamma, t_shift)

            # (B) Protocol invariance under realistic confounds (still paired across TR):
            # vb_list = parse_float_list2(args.vb_list)
            # tsh_list = parse_float_list2(args.t_shift_list)
            # gam_list = parse_float_list2(args.gamma_list)
            # def sample_confounds(_rng):
            #     return float(_rng.choice(gam_list)), float(_rng.choice(tsh_list))

            for aif_mode in (["bolus","infusion"] if args.aif_mode == "both" else [args.aif_mode]):

                for match_id in range(int(args.n_per_cell)):
                    # --- Sample "true biology" once, then reuse across all TRs ---
                    vb0 = float(rng.choice(vb_grid))      # you already define vb_grid
                    kpl0 = float(rng.choice(kpl_grid))    # you already define kpl_grid
                    gam0, tsh0 = sample_confounds(rng)

                    # Keep same kinetic truth across protocols
                    pars_true = KineticParams(
                        kpl=kpl0, kve=args.kve, vb=vb0,
                        r1p=cfg.r1p, r1l=cfg.r1l
                    )

                    for TR in tr_list:
                        # constant window: choose n so t[-1] ≈ t_end but not exceeding it
                        n_tp = int(np.floor(t_end / TR)) + 1
                        cfg_tr = cfg_with(cfg, TR=float(TR), n_timepoints=int(n_tp))

                        sim = simulate_one(
                            cfg_tr, pars_true,
                            aif_mode=aif_mode,
                            t_shift=float(tsh0),
                            gamma_gain=float(gam0),
                            delta_theta_p_deg=0.0,
                            delta_theta_l_deg=0.0,
                            rng=set_seed(np.random.randint(1_000_000_000))
                        )

                        # Estimate kPL_hat using your existing helper
                        kpl_hat = estimate_kpl_hat(
                            sim, cfg_tr, args, nn_ctx,
                            init_kpl=float(kpl0),
                            init_vb=float(vb0),
                            r1p=float(cfg_tr.r1p),
                            r1l=float(cfg_tr.r1l)
                        )

                        rows.append({
                            "experiment": "trsweep",
                            "match_id": int(match_id),
                            "aif_mode": aif_mode,
                            "TR": float(TR),
                            "n_timepoints": int(n_tp),
                            "t_end_eff": float((n_tp - 1) * TR),

                            "t_shift": float(tsh0),
                            "gamma": float(gam0),

                            "kpl_true": float(kpl0),
                            "kve_true": float(args.kve),
                            "vb_true": float(vb0),
                            "r1p_true": float(cfg_tr.r1p),
                            "r1l_true": float(cfg_tr.r1l),
                            "flip_delta_p": 0.0,
                            "flip_delta_l": 0.0,
                            "snr_pyr": float(cfg_tr.snr_pyr),
                            "snr_lac": float(cfg_tr.snr_lac),

                            "auc_ratio": float(sim["auc_ratio"]),
                            "kpl_hat": float(kpl_hat),
                        })

            # done with trsweep
            # return rows
            continue
        
        # ---- aifshape: delivery-shape + arrival variability sweep ----
        if args.experiment == "aifshape":
            # parse grids
            vb_list = parse_float_list2(args.vb_list) if hasattr(args, "vb_list") else list(vb_grid)
            kpl_list = parse_float_list2(args.kpl_list) if hasattr(args, "kpl_list") else list(kpl_grid)
            tsh_list = parse_float_list2(args.t_shift_list)  # reuse existing t_shift_list

            aif_alpha_list = parse_float_list2(args.aif_alpha_list)
            aif_beta_list  = parse_float_list2(args.aif_beta_list)
            aif_t0_list    = parse_float_list2(args.aif_t0_list)
            aif_tau_list   = parse_float_list2(args.aif_tau_list)

            for aif_mode in (["bolus","infusion"] if args.aif_mode == "both" else [args.aif_mode]):

                # Choose which shape params to sweep based on mode
                if aif_mode == "bolus":
                    shape_grid = [(a, b, t0, 20.0) for a in aif_alpha_list for b in aif_beta_list for t0 in aif_t0_list]
                else:  # infusion
                    shape_grid = [(2.0, 6.0, t0, tau) for t0 in aif_t0_list for tau in aif_tau_list]

                for (aif_alpha, aif_beta, aif_t0, aif_tau) in shape_grid:
                    for t_shift in tsh_list:
                        for vb in vb_list:
                            for kpl in kpl_list:
                                for _ in range(args.n_per_cell):
                                    pars_true = KineticParams(kpl=float(kpl), kve=args.kve, vb=float(vb),
                                                            r1p=cfg.r1p, r1l=cfg.r1l)

                                    sim = simulate_one(
                                        cfg, pars_true,
                                        aif_mode=aif_mode,
                                        t_shift=float(t_shift),
                                        gamma_gain=1.0,
                                        delta_theta_p_deg=0.0,
                                        delta_theta_l_deg=0.0,
                                        aif_alpha=float(aif_alpha),
                                        aif_beta=float(aif_beta),
                                        aif_t0=float(aif_t0),
                                        aif_tau=float(aif_tau),
                                        rng=set_seed(np.random.randint(1e9))
                                    )

                                    kpl_hat = np.nan
                                    if args.estimate == "measured_driver":
                                        bounds = ([0.0,0.0,0.005],[0.5,1.0,0.20])
                                        init = (float(kpl), args.kve, float(vb))
                                        res = fit_measured_driver_lactate(
                                            sim["t"], sim["S_l"], sim["U_used"],
                                            sim["theta_p"], sim["theta_l"], cfg.TR,
                                            init, bounds, cfg.r1p, cfg.r1l
                                        )
                                        kpl_hat = res["kpl_hat"]

                                    elif args.estimate == "joint":
                                        bounds = ([0.0,0.0,0.005],[0.5,1.0,0.20])
                                        init = (float(kpl), args.kve, float(vb))
                                        res = fit_joint_pyruvate_lactate(
                                            sim["t"], sim["S_p"], sim["S_l"],
                                            sim["theta_p"], sim["theta_l"], cfg.TR,
                                            init, bounds, cfg.r1p, cfg.r1l
                                        )
                                        kpl_hat = res["kpl_hat"]

                                    elif args.estimate == "nn":
                                        kpl_hat = nn_ctx["predict_kpl"](sim["S_p"], sim["S_l"])

                                    rows.append({
                                        "aif_mode": aif_mode,
                                        "t_shift": float(t_shift),
                                        "gamma": 1.0,

                                        "aif_alpha": float(sim["aif_alpha"]),
                                        "aif_beta":  float(sim["aif_beta"]),
                                        "aif_t0":    float(sim["aif_t0"]),
                                        "aif_tau":   float(sim["aif_tau"]),

                                        "kpl_true": float(kpl),
                                        "kve_true": float(args.kve),
                                        "vb_true": float(vb),

                                        "r1p_true": float(cfg.r1p),
                                        "r1l_true": float(cfg.r1l),

                                        "flip_delta_p": 0.0,
                                        "flip_delta_l": 0.0,

                                        "snr_pyr": float(cfg.snr_pyr),
                                        "snr_lac": float(cfg.snr_lac),

                                        "auc_ratio": float(sim["auc_ratio"]),
                                        "kpl_hat": float(kpl_hat),
                                    })

            continue

        # inside run_experiment(), after you've built vb_grid/t_shift_list/gamma_grid, etc.
        if args.experiment == "matchratio":
            resample = set([x.strip() for x in args.match_resample.split(",") if x.strip()])

            vb_list = parse_float_list2(args.vb_list)
            tsh_list = parse_float_list2(args.t_shift_list)
            gam_list = parse_float_list2(args.gamma_list)

            def sample_confounds(rng):
                vb = float(rng.choice(vb_list)) if ("vb" in resample) else float(vb_grid[0])
                tsh = float(rng.choice(tsh_list)) if ("tshift" in resample) else float(0.0)
                gam = float(rng.choice(gam_list)) if ("gamma" in resample) else float(1.0)
                return vb, tsh, gam

            rng = np.random.default_rng(args.seed)
            for aif_mode in (["bolus","infusion"] if args.aif_mode == "both" else [args.aif_mode]):

                for match_id in range(args.n_per_cell):
                    # --- NEG sample ---
                    vb0, tsh0, gam0 = sample_confounds(rng)
                    pars_neg = KineticParams(kpl=args.roc_kpl_neg, kve=args.kve, vb=vb0, r1p=cfg.r1p, r1l=cfg.r1l)
                    sim_neg = simulate_one(cfg, pars_neg, aif_mode=aif_mode,
                                        t_shift=tsh0, gamma_gain=gam0,
                                        delta_theta_p_deg=0.0, delta_theta_l_deg=0.0,
                                        rng=set_seed(np.random.randint(1e9)))
                    target = float(sim_neg["auc_ratio"])

                    # --- POS sample: resample confounds until auc_ratio matches target ---
                    sim_pos = None
                    vb1=tsh1=gam1=None
                    for _ in range(args.match_max_tries):
                        vb1, tsh1, gam1 = sample_confounds(rng)
                        pars_pos = KineticParams(kpl=args.roc_kpl_pos, kve=args.kve, vb=vb1, r1p=cfg.r1p, r1l=cfg.r1l)
                        cand = simulate_one(cfg, pars_pos, aif_mode=aif_mode,
                                            t_shift=tsh1, gamma_gain=gam1,
                                            delta_theta_p_deg=0.0, delta_theta_l_deg=0.0,
                                            rng=set_seed(np.random.randint(1e9)))
                        if np.isfinite(target) and np.isfinite(cand["auc_ratio"]) and abs(float(cand["auc_ratio"]) - target) <= args.match_tol:
                            sim_pos = cand
                            break

                    if sim_pos is None:
                        continue

                    # estimate kpl_hat for both (reuses your estimator code through the helper)
                    kpl_hat_neg = estimate_kpl_hat(sim_neg, cfg, args, nn_ctx,
                                                init_kpl=args.roc_kpl_neg, init_vb=vb0, r1p=cfg.r1p, r1l=cfg.r1l)
                    kpl_hat_pos = estimate_kpl_hat(sim_pos, cfg, args, nn_ctx,
                                                init_kpl=args.roc_kpl_pos, init_vb=vb1, r1p=cfg.r1p, r1l=cfg.r1l)

                    rows.append({
                        "experiment": "matchratio",
                        "y": 0, "match_id": int(match_id),
                        "aif_mode": aif_mode, "t_shift": float(tsh0), "gamma": float(gam0),
                        "kpl_true": float(args.roc_kpl_neg), "kve_true": float(args.kve), "vb_true": float(vb0),
                        "r1p_true": float(cfg.r1p), "r1l_true": float(cfg.r1l),
                        "flip_delta_p": 0.0, "flip_delta_l": 0.0,
                        "snr_pyr": float(cfg.snr_pyr), "snr_lac": float(cfg.snr_lac),
                        "auc_ratio": float(sim_neg["auc_ratio"]), "kpl_hat": float(kpl_hat_neg),
                    })
                    rows.append({
                        "experiment": "matchratio",
                        "y": 1, "match_id": int(match_id),
                        "aif_mode": aif_mode, "t_shift": float(tsh1), "gamma": float(gam1),
                        "kpl_true": float(args.roc_kpl_pos), "kve_true": float(args.kve), "vb_true": float(vb1),
                        "r1p_true": float(cfg.r1p), "r1l_true": float(cfg.r1l),
                        "flip_delta_p": 0.0, "flip_delta_l": 0.0,
                        "snr_pyr": float(cfg.snr_pyr), "snr_lac": float(cfg.snr_lac),
                        "auc_ratio": float(sim_pos["auc_ratio"]), "kpl_hat": float(kpl_hat_pos),
                    })

            continue
        
        # ---- AUROC experiment ----
        if args.experiment == "roc":
            if args.roc_group == "gamma":
                groups = [("gamma", float(g)) for g in gamma_list_roc]
            elif args.roc_group == "vb":
                groups = [("vb_true", float(v)) for v in vb_list_roc]
            elif args.roc_group == "tshift":
                groups = [("t_shift", float(sh)) for sh in t_shift_list]
            else:
                groups = [("all", 0.0)]

            for (gkey, gval) in groups:
                gamma_val = gval if gkey=="gamma" else 1.0
                vb_val    = gval if gkey=="vb_true" else 0.02
                tsh_val   = gval if gkey=="t_shift" else 0.0
                n_pos = int(round(args.n_per_cell * args.roc_pos_frac))
                n_neg = int(args.n_per_cell - n_pos)
                labels = np.array([1]*n_pos + [0]*n_neg)
                rng.shuffle(labels)
                for lab in labels:
                    kpl = args.roc_kpl_pos if lab==1 else args.roc_kpl_neg
                    pars_true = KineticParams(kpl=kpl, kve=args.kve, vb=vb_val, r1p=cfg.r1p, r1l=cfg.r1l)
                    sim = simulate_one(cfg, pars_true, aif_mode=aif_mode,
                                       t_shift=tsh_val, gamma_gain=gamma_val,
                                       delta_theta_p_deg=0.0, delta_theta_l_deg=0.0,
                                       rng=set_seed(np.random.randint(1e9)))
                    # estimator selection
                    kpl_hat = np.nan
                    if args.estimate == "measured_driver":
                        bounds = ([0.0,0.0,0.005],[0.5,1.0,0.20]); init=(kpl, args.kve, vb_val)
                        res = fit_measured_driver_lactate(sim["t"], sim["S_l"], sim["U_used"],
                                                          sim["theta_p"], sim["theta_l"], cfg.TR,
                                                          init, bounds, cfg.r1p, cfg.r1l)
                        kpl_hat = res["kpl_hat"]
                    elif args.estimate == "joint":
                        bounds = ([0.0,0.0,0.005],[0.5,1.0,0.20]); init=(kpl, args.kve, vb_val)
                        res = fit_joint_pyruvate_lactate(sim["t"], sim["S_p"], sim["S_l"],
                                                         sim["theta_p"], sim["theta_l"], cfg.TR,
                                                         init, bounds, cfg.r1p, cfg.r1l)
                        kpl_hat = res["kpl_hat"]
                    elif args.estimate == "nn":
                        kpl_hat = nn_ctx["predict_kpl"](sim["S_p"], sim["S_l"])

                    rows.append({
                        "aif_mode": aif_mode,
                        "t_shift": float(tsh_val),
                        "gamma": float(gamma_val),
                        "kpl_true": float(kpl),
                        "kve_true": float(args.kve),
                        "vb_true": float(vb_val),
                        "r1p_true": float(cfg.r1p),
                        "r1l_true": float(cfg.r1l),
                        "flip_delta_p": 0.0,
                        "flip_delta_l": 0.0,
                        "snr_pyr": float(cfg.snr_pyr),
                        "snr_lac": float(cfg.snr_lac),
                        "auc_ratio": float(sim["auc_ratio"]),
                        "kpl_hat": float(kpl_hat),
                        "is_pos": int(lab),
                        "roc_group": gval if gkey != "all" else "all"
                    })
            continue

        # ---- tshift sweep ----
        if args.experiment == "tshift":
            for t_shift in t_shift_list:
                for vb in vb_grid:
                    for kpl in kpl_grid:
                        for _ in range(args.n_per_cell):
                            pars_true = KineticParams(kpl=kpl, kve=args.kve, vb=vb, r1p=cfg.r1p, r1l=cfg.r1l)
                            sim = simulate_one(cfg, pars_true, aif_mode=aif_mode,
                                               t_shift=float(t_shift), gamma_gain=1.0,
                                               delta_theta_p_deg=0.0, delta_theta_l_deg=0.0,
                                               rng=set_seed(np.random.randint(1e9)))
                            kpl_hat = np.nan
                            if args.estimate == "measured_driver":
                                bounds=([0.0,0.0,0.005],[0.5,1.0,0.20]); init=(kpl,args.kve,vb)
                                res = fit_measured_driver_lactate(sim["t"], sim["S_l"], sim["U_used"],
                                                                  sim["theta_p"], sim["theta_l"], cfg.TR,
                                                                  init, bounds, cfg.r1p, cfg.r1l)
                                kpl_hat = res["kpl_hat"]
                            elif args.estimate == "joint":
                                bounds=([0.0,0.0,0.005],[0.5,1.0,0.20]); init=(kpl,args.kve,vb)
                                res = fit_joint_pyruvate_lactate(sim["t"], sim["S_p"], sim["S_l"],
                                                                 sim["theta_p"], sim["theta_l"], cfg.TR,
                                                                 init, bounds, cfg.r1p, cfg.r1l)
                                kpl_hat = res["kpl_hat"]
                            elif args.estimate == "nn":
                                kpl_hat = nn_ctx["predict_kpl"](sim["S_p"], sim["S_l"])
                            rows.append({
                                "aif_mode": aif_mode, "t_shift": float(t_shift), "gamma": 1.0,
                                "kpl_true": float(kpl), "kve_true": float(args.kve), "vb_true": float(vb),
                                "r1p_true": float(cfg.r1p), "r1l_true": float(cfg.r1l),
                                "flip_delta_p": 0.0, "flip_delta_l": 0.0,
                                "snr_pyr": float(cfg.snr_pyr), "snr_lac": float(cfg.snr_lac),
                                "auc_ratio": float(sim["auc_ratio"]), "kpl_hat": float(kpl_hat),
                            })
            continue

        # ---- t1flip ----
        for t_shift in t_shift_grid:
            if args.experiment == "t1flip":
                for T1p in t1p_list:
                    for T1l in t1l_list:
                        r1p = 1.0/max(T1p,1e-6); r1l = 1.0/max(T1l,1e-6)
                        for dthp in dthp_list:
                            for dthl in dthl_list:
                                for kpl in kpl_grid:
                                    for vb in vb_grid:
                                        for _ in range(args.n_per_cell):
                                            pars_true = KineticParams(kpl=kpl,kve=args.kve,vb=vb,r1p=r1p,r1l=r1l)
                                            sim = simulate_one(cfg, pars_true, aif_mode=aif_mode, t_shift=t_shift,
                                                               gamma_gain=1.0, delta_theta_p_deg=dthp, delta_theta_l_deg=dthl,
                                                               rng=set_seed(np.random.randint(1e9)))
                                            kpl_hat = np.nan
                                            if args.estimate == "measured_driver":
                                                bounds=([0.0,0.0,0.005],[0.5,1.0,0.20]); init=(kpl,args.kve,vb)
                                                res = fit_measured_driver_lactate(sim["t"], sim["S_l"], sim["U_used"],
                                                                                  sim["theta_p"], sim["theta_l"], cfg.TR,
                                                                                  init, bounds, r1p, r1l)
                                                kpl_hat = res["kpl_hat"]
                                            elif args.estimate == "joint":
                                                bounds=([0.0,0.0,0.005],[0.5,1.0,0.20]); init=(kpl,args.kve,vb)
                                                res = fit_joint_pyruvate_lactate(sim["t"], sim["S_p"], sim["S_l"],
                                                                                 sim["theta_p"], sim["theta_l"], cfg.TR,
                                                                                 init, bounds, r1p, r1l)
                                                kpl_hat = res["kpl_hat"]
                                            elif args.estimate == "nn":
                                                kpl_hat = nn_ctx["predict_kpl"](sim["S_p"], sim["S_l"])
                                            rows.append({
                                                "aif_mode": aif_mode,"t_shift": float(t_shift),"gamma": 1.0,
                                                "kpl_true": float(kpl),"kve_true": float(args.kve),"vb_true": float(vb),
                                                "r1p_true": float(r1p),"r1l_true": float(r1l),
                                                "flip_delta_p": float(dthp),"flip_delta_l": float(dthl),
                                                "snr_pyr": float(cfg.snr_pyr),"snr_lac": float(cfg.snr_lac),
                                                "auc_ratio": float(sim["auc_ratio"]), "kpl_hat": float(kpl_hat),
                                            })
                continue

            # ---- gain/vb/kpl ----
            for gamma in gamma_grid:
                for vb in vb_grid:
                    for kpl in kpl_grid:
                        for _ in range(args.n_per_cell):
                            pars_true = KineticParams(kpl=kpl,kve=args.kve,vb=vb,r1p=cfg.r1p,r1l=cfg.r1l)
                            sim = simulate_one(cfg, pars_true, aif_mode=aif_mode, t_shift=t_shift, gamma_gain=gamma,
                                               delta_theta_p_deg=0.0, delta_theta_l_deg=0.0,
                                               rng=set_seed(np.random.randint(1e9)))
                            kpl_hat = np.nan
                            if args.estimate == "measured_driver":
                                bounds=([0.0,0.0,0.005],[0.5,1.0,0.20]); init=(kpl,args.kve,vb)
                                res = fit_measured_driver_lactate(sim["t"], sim["S_l"], sim["U_used"],
                                                                  sim["theta_p"], sim["theta_l"], cfg.TR,
                                                                  init, bounds, cfg.r1p, cfg.r1l)
                                kpl_hat = res["kpl_hat"]
                            elif args.estimate == "joint":
                                bounds=([0.0,0.0,0.005],[0.5,1.0,0.20]); init=(kpl,args.kve,vb)
                                res = fit_joint_pyruvate_lactate(sim["t"], sim["S_p"], sim["S_l"],
                                                                 sim["theta_p"], sim["theta_l"], cfg.TR,
                                                                 init, bounds, cfg.r1p, cfg.r1l)
                                kpl_hat = res["kpl_hat"]
                            elif args.estimate == "nn":
                                kpl_hat = nn_ctx["predict_kpl"](sim["S_p"], sim["S_l"])
                            rows.append({
                                "aif_mode": aif_mode,"t_shift": float(t_shift),"gamma": float(gamma),
                                "kpl_true": float(kpl),"kve_true": float(args.kve),"vb_true": float(vb),
                                "r1p_true": float(cfg.r1p),"r1l_true": float(cfg.r1l),
                                "flip_delta_p": 0.0,"flip_delta_l": 0.0,
                                "snr_pyr": float(cfg.snr_pyr),"snr_lac": float(cfg.snr_lac),
                                "auc_ratio": float(sim["auc_ratio"]), "kpl_hat": float(kpl_hat),
                            })

    if not rows: raise RuntimeError("No rows generated; check arguments.")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    # ----- summary metrics -----
    y_lp = np.array([r["auc_ratio"] for r in rows], float)
    y_kp = np.array([r.get("kpl_hat", np.nan) for r in rows], float)
    kpl_true = np.array([r["kpl_true"] for r in rows], float)
    gamma_arr = np.array([r["gamma"] for r in rows], float)
    vb_arr = np.array([r["vb_true"] for r in rows], float)
    tshift_arr = np.array([r["t_shift"] for r in rows], float)
    mode_inf = np.array([1.0 if r["aif_mode"]=="infusion" else 0.0 for r in rows], float)
    r1p_arr = np.array([r.get("r1p_true", np.nan) for r in rows], float)
    r1l_arr = np.array([r.get("r1l_true", np.nan) for r in rows], float)
    dthp_arr = np.array([r.get("flip_delta_p", 0.0) for r in rows], float)
    dthl_arr = np.array([r.get("flip_delta_l", 0.0) for r in rows], float)

    cov_lp_all = cov(y_lp); cov_kp_all = cov(y_kp)

    def elasticity_by_gamma(metric):
        vals, xs = [], []
        for g in sorted(set(gamma_arr)):
            ys = metric[gamma_arr==g]; ys = ys[np.isfinite(ys)]
            if ys.size>0: vals.append(np.nanmean(ys)); xs.append(g)
        return loglog_slope(np.array(xs), np.array(vals)) if len(xs)>=2 else np.nan

    elas_lp = elasticity_by_gamma(y_lp) if args.experiment=="gain" else np.nan
    elas_kp = elasticity_by_gamma(y_kp) if args.experiment=="gain" else np.nan

    r2_lp = r2_linreg(y_lp, kpl_true); r2_kp = r2_linreg(y_kp, kpl_true)
    r2corr_lp = r2_corr(y_lp, kpl_true); r2corr_kp = r2_corr(y_kp, kpl_true)

    conf = np.column_stack([gamma_arr, mode_inf, vb_arr, tshift_arr, r1p_arr, r1l_arr, dthp_arr, dthl_arr])
    ok_lp = np.isfinite(y_lp) & np.all(np.isfinite(conf), axis=1) & np.isfinite(kpl_true)
    ok_kp = np.isfinite(y_kp) & np.all(np.isfinite(conf), axis=1) & np.isfinite(kpl_true)
    pr2_lp = partial_r2(y_lp[ok_lp], np.column_stack([np.ones(ok_lp.sum()), conf[ok_lp]]),
                        np.column_stack([np.ones(ok_lp.sum()), conf[ok_lp], kpl_true[ok_lp].reshape(-1,1)])) if ok_lp.any() else np.nan
    pr2_kp = partial_r2(y_kp[ok_kp], np.column_stack([np.ones(ok_kp.sum()), conf[ok_kp]]),
                        np.column_stack([np.ones(ok_kp.sum()), conf[ok_kp], kpl_true[ok_kp].reshape(-1,1)])) if ok_kp.any() else np.nan

    # AUROC
    auc_overall_lp = auc_overall_kp = np.nan
    auc_by_group_lp, auc_by_group_kp = {}, {}
    if args.experiment == "roc":
        y_true = np.array([r["is_pos"] for r in rows], int)
        auc_overall_lp = auc_from_scores(y_true, y_lp)
        auc_overall_kp = auc_from_scores(y_true, y_kp)
        grp_vals = [r["roc_group"] for r in rows]
        for g in sorted(set(grp_vals), key=lambda x: (isinstance(x, str), x)):
            m = np.array([gg==g for gg in grp_vals], bool)
            auc_by_group_lp[str(g)] = auc_from_scores(y_true[m], y_lp[m])
            auc_by_group_kp[str(g)] = auc_from_scores(y_true[m], y_kp[m])

    summary = {
        "n_rows": len(rows),
        "experiment": args.experiment,
        "aif_mode": args.aif_mode,
        "schedule": args.schedule,
        "metrics": {
            "cov_LacPyr_overall": cov_lp_all,
            "cov_kPLhat_overall": cov_kp_all,
            "elasticity_LacPyr_vs_gamma": elas_lp,
            "elasticity_kPLhat_vs_gamma": elas_kp,
            "olsR2_LacPyr_on_true_kPL": r2_lp,
            "olsR2_kPLhat_on_true_kPL": r2_kp,
            "corr2_LacPyr_vs_true_kPL": r2corr_lp,
            "corr2_kPLhat_vs_true_kPL": r2corr_kp,
            "partialR2_LacPyr_given_confounds": pr2_lp,
            "partialR2_kPLhat_given_confounds": pr2_kp,
            "auc_overall_LacPyr": auc_overall_lp,
            "auc_overall_kPLhat": auc_overall_kp,
        }
    }
    print("\n=== SUMMARY ==="); print(json.dumps(summary, indent=2))
    if args.summary_out:
        with open(args.summary_out, "w") as f: json.dump(summary, f, indent=2)

    write_stats_csv_twolevel(args.stats_out, rows, args.experiment, min_rep=10)


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="HP-13C enriched simulator with NN estimator, AUROC & multiple experiments")
    p.add_argument("--out", required=True); p.add_argument("--summary_out", default=""); p.add_argument("--stats_out", default="")
    p.add_argument("--n_per_cell", type=int, default=40); p.add_argument("--seed", type=int, default=1337)
    # acquisition
    p.add_argument("--TR", type=float, default=2.0); p.add_argument("--n_timepoints", type=int, default=30)
    p.add_argument("--schedule", choices=["VFA","CFA"], default="VFA")
    p.add_argument("--cfa_pyr_deg", type=float, default=11.0); p.add_argument("--cfa_lac_deg", type=float, default=80.0)
    p.add_argument("--snr_pyr", type=float, default=32.0); p.add_argument("--snr_lac", type=float, default=20.0)
    p.add_argument("--r1p", type=float, default=1/30.0); p.add_argument("--r1l", type=float, default=1/25.0)
    # kinetics
    p.add_argument("--kve", type=float, default=0.25)
    # AIF/timing
    p.add_argument("--aif_mode", choices=["bolus","infusion","both"], default="bolus")
    p.add_argument("--t_shift", type=float, default=0.0); p.add_argument("--t_shift_list", type=str, default="-10,-5,0,5,10")
    # T1/flip lists
    p.add_argument("--t1p_list", type=str, default=""); p.add_argument("--t1l_list", type=str, default="")
    p.add_argument("--flip_delta_p_list", type=str, default=""); p.add_argument("--flip_delta_l_list", type=str, default="")
    # ROC-specific options
    p.add_argument("--roc_kpl_neg", type=float, default=0.12, help="kPL for negative class")
    p.add_argument("--roc_kpl_pos", type=float, default=0.144, help="kPL for positive class")
    p.add_argument("--roc_pos_frac", type=float, default=0.5, help="Fraction of positives in each group")
    p.add_argument("--roc_group", choices=["gamma","vb","tshift","none"], default="gamma", help="Group axis for AUROC")
    p.add_argument("--gamma_list", type=str, default="0.5,0.75,1.0,1.25,1.5", help="Gamma values for roc gain sweep")
    p.add_argument("--vb_list", type=str, default="0.01,0.02,0.04,0.06,0.08", help="vB values for roc vb sweep")
    # AIF-shape experiment grids (bolus + infusion)
    p.add_argument("--aif_alpha_list", type=str, default="1.5,2.0,2.5,3.0",
                help="Gamma-variate alpha values for AIF-shape experiment (bolus).")
    p.add_argument("--aif_beta_list", type=str, default="4.0,6.0,8.0",
                help="Gamma-variate beta values for AIF-shape experiment (bolus).")
    p.add_argument("--aif_t0_list", type=str, default="1.0,2.0,3.0",
                help="Bolus/infusion arrival t0 values (s) for AIF-shape experiment.")
    p.add_argument("--aif_tau_list", type=str, default="10.0,20.0,30.0",
                help="Infusion tau values (s) for AIF-shape experiment (infusion).")
    # estimator
    p.add_argument("--estimate", choices=["none","measured_driver","joint","nn"], default="none")
    # NN options
    p.add_argument("--nn_weights", type=str, default="", help="Path to HybridMultiHead .pth/.pt weights")
    p.add_argument("--nn_kpl_index", type=int, default=0, help="Index of kPL in NN output vector (default 0)")
    p.add_argument("--nn_ptrain", type=float, default=1.0, help="Training-time robust peak reference (P_train)")
    p.add_argument("--nn_use_robust_peak", action="store_true", help="Apply robust-peak normalization before NN (recommended)")
    
    # add to parser
    p.add_argument("--experiment", choices=["gain","vb","kpl","t1flip","tshift","roc","matchratio","aifshape","trsweep"], required=True)
    # TR sweep (constant window)
    p.add_argument("--tr_list", type=str, default="2,3,5", help="Comma list of TRs in seconds for trsweep.")
    p.add_argument("--t_end", type=float, default=30.0, help="Constant acquisition window end time (s) for trsweep.")



    # matching controls
    p.add_argument("--match_metric", choices=["auc_ratio"], default="auc_ratio")
    p.add_argument("--match_tol", type=float, default=0.01, help="Absolute tolerance for matching Lac/Pyr AUC ratio")
    p.add_argument("--match_max_tries", type=int, default=2000, help="How many resamples to try to match")
    p.add_argument("--match_resample", type=str, default="gamma,vb,tshift",
                help="Comma-separated confounds to resample to match Lac/Pyr (e.g. gamma,vb,tshift)")


    p.add_argument("--ratio_mode", choices=["auc_full","auc_window","tpoint"], default="auc_full")
    p.add_argument("--ratio_tmin", type=float, default=0.0)
    p.add_argument("--ratio_tmax", type=float, default=60.0)
    p.add_argument("--ratio_tpoint", type=float, default=30.0)

    
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
