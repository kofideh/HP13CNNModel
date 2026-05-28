#!/usr/bin/env python3
"""
Corrected Figure 5 matched-ratio simulation for HP-13C pyruvate/lactate kinetics.

Purpose
-------
This script replaces the previous matched-ratio experiment that used a
"pyruvate-only receive gain" nuisance factor. In this corrected version,
positive and negative samples are matched on Lac/Pyr AUC ratio by resampling
physiologically defensible confounds:

  - vascular volume fraction vB
  - vascular-extravascular exchange kVE
  - pyruvate bolus timing / acquisition-window shift
  - pyruvate driver/AIF shape parameters
  - independent noise realizations

No metabolite-specific receive-gain term is used for Figure 5.

Corrected kinetic model
-----------------------
The simulated pyruvate curve is treated as the measured pyruvate driver
S_pyr(t). Starting from the corrected exchange model,

  dPe/dt = kVE Pv(t) - [kVE + kPL + R1p_app(t)] Pe(t)
  S_pyr(t) = vB Pv(t) + (1-vB) Pe(t),

and eliminating intravascular pyruvate gives:

  dPe/dt = (kVE/vB) S_pyr(t)
          - [kVE/vB + kPL + R1p_app(t)] Pe(t)

  dL/dt  = kPL Pe(t) - R1l_app(t) L(t)

Observed lactate is S_lac(t) = (1-vB) L(t).

This matches the corrected measured-driver model and avoids explicitly
prescribing a separate metabolite-specific receive gain.
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility wrapper for trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def parse_float_list(text: str, default: List[float]) -> List[float]:
    if text is None or str(text).strip() == "":
        return list(default)
    return [float(v.strip()) for v in str(text).split(",") if v.strip()]


def parse_range(text: str, default: Tuple[float, float]) -> Tuple[float, float]:
    vals = parse_float_list(text, list(default))
    if len(vals) != 2:
        raise ValueError(f"Expected two comma-separated values, got: {text}")
    lo, hi = float(vals[0]), float(vals[1])
    if hi <= lo:
        raise ValueError(f"Upper bound must exceed lower bound, got: {text}")
    return lo, hi


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


# -----------------------------------------------------------------------------
# Acquisition and signal-generation utilities
# -----------------------------------------------------------------------------


def vfa_schedule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric VFA-like schedule used in the original script."""
    th_p, th_l = [], []
    for i in range(n):
        val = math.atan(1.0 / math.sqrt(max(1, 16 - i))) if i < n - 1 else math.radians(90.0)
        th_p.append(val)
        th_l.append(val)
    return np.array(th_p, dtype=float), np.array(th_l, dtype=float)


def cfa_schedule(n: int, pyr_deg: float, lac_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.full(n, math.radians(float(pyr_deg)), dtype=float),
        np.full(n, math.radians(float(lac_deg)), dtype=float),
    )


def r1_app(r1: float, theta: np.ndarray, tr: float) -> np.ndarray:
    theta = np.clip(theta, 0.0, math.radians(89.9))
    return float(r1) - np.log(np.cos(theta)) / float(tr)


def aif_gamma(t: np.ndarray, A: float, t0: float, alpha: float, beta: float) -> np.ndarray:
    tp = np.maximum(t - float(t0), 0.0)
    return float(A) * (tp ** float(alpha)) * np.exp(-tp / float(beta))


def aif_infusion(t: np.ndarray, A: float, t0: float, tau: float, ramp: bool = False) -> np.ndarray:
    tp = t - float(t0)
    if ramp:
        x = np.clip(1.0 - np.abs((tp - float(tau) / 2.0) / (float(tau) / 2.0)), 0.0, 1.0)
        return float(A) * x
    return float(A) * ((tp >= 0.0) & (tp <= float(tau))).astype(float)


def normalize_area(y: np.ndarray, t: np.ndarray, target_area: float) -> np.ndarray:
    area = trapz(y, t)
    return y * (float(target_area) / max(area, 1e-12))


@dataclass
class SimConfig:
    TR: float = 2.0
    n_timepoints: int = 16
    schedule: str = "VFA"
    cfa_pyr_deg: float = 11.0
    cfa_lac_deg: float = 80.0
    snr_pyr: float = 32.0
    snr_lac: float = 20.0
    r1p: float = 1.0 / 30.0
    r1l: float = 1.0 / 25.0


@dataclass
class KineticParams:
    kpl: float
    kve: float
    vb: float
    r1p: float
    r1l: float


@dataclass
class DriverParams:
    aif_mode: str = "bolus"
    t_shift: float = 0.0
    aif_alpha: float = 2.0
    aif_beta: float = 6.0
    aif_t0: float = 2.0
    aif_tau: float = 20.0


def build_measured_pyruvate_driver(cfg: SimConfig, drv: DriverParams) -> Tuple[np.ndarray, np.ndarray]:
    """Return acquisition-time vector and clean measured S_pyr(t) driver."""
    n = int(cfg.n_timepoints)
    t = np.arange(n, dtype=float) * float(cfg.TR)
    tf = np.linspace(0.0, max(float(t[-1]), 1e-6), 2001)

    # Keep delivered pyruvate-driver area fixed across shape perturbations.
    ref = aif_gamma(tf, A=1.0, t0=2.0, alpha=2.0, beta=6.0)
    dose_ref = trapz(ref, tf)

    mode = drv.aif_mode.lower()
    if mode == "bolus":
        y = aif_gamma(tf, A=1.0, t0=drv.aif_t0, alpha=drv.aif_alpha, beta=drv.aif_beta)
    elif mode == "infusion":
        y = aif_infusion(tf, A=1.0, t0=drv.aif_t0, tau=drv.aif_tau, ramp=False)
    else:
        raise ValueError("aif_mode must be 'bolus' or 'infusion'.")

    y = normalize_area(y, tf, dose_ref)

    # Positive t_shift delays the bolus relative to the acquisition window.
    y_shifted = np.interp(tf - float(drv.t_shift), tf, y, left=0.0, right=0.0)
    S_pyr_driver = np.interp(t, tf, y_shifted, left=0.0, right=0.0)
    return t, S_pyr_driver


def integrate_measured_driver(
    t: np.ndarray,
    S_pyr_driver: np.ndarray,
    theta_p: np.ndarray,
    theta_l: np.ndarray,
    tr: float,
    pars: KineticParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate corrected measured-driver model using exact interval updates.

    S_pyr_driver is the measured pyruvate signal S_pyr(t), not a separate
    intravascular concentration with a metabolite-specific receive gain.
    """
    t = np.asarray(t, dtype=float)
    S_pyr_driver = np.asarray(S_pyr_driver, dtype=float)
    R1p_app = r1_app(pars.r1p, theta_p, tr)
    R1l_app = r1_app(pars.r1l, theta_l, tr)

    n = len(t)
    Pe = np.zeros(n, dtype=float)
    L = np.zeros(n, dtype=float)

    for i in range(1, n):
        dt = float(t[i] - t[i - 1])
        U = float(S_pyr_driver[i - 1])
        # Correct measured-driver equation after eliminating Pv from:
        #   dPe/dt = kVE*Pv - (kVE + kPL + R1p_app)*Pe
        #   S_pyr = vB*Pv + (1-vB)*Pe
        # gives:
        #   dPe/dt = (kVE/vB)*S_pyr - (kVE/vB + kPL + R1p_app)*Pe
        a = (pars.kve / pars.vb) + pars.kpl + float(R1p_app[i - 1])
        bU = (pars.kve / pars.vb) * U
        c = float(R1l_app[i - 1])

        a = max(a, 1e-12)
        c = max(c, 1e-12)
        Pe0 = float(Pe[i - 1])
        L0 = float(L[i - 1])
        Pe_ss = bU / a
        exp_a = math.exp(-a * dt)
        exp_c = math.exp(-c * dt)

        Pe[i] = Pe_ss + (Pe0 - Pe_ss) * exp_a

        if abs(c - a) < 1e-10:
            transient_integral = (Pe0 - Pe_ss) * dt * exp_c
        else:
            transient_integral = (Pe0 - Pe_ss) * (exp_a - exp_c) / (c - a)
        steady_integral = Pe_ss * (1.0 - exp_c) / c
        L[i] = L0 * exp_c + pars.kpl * (steady_integral + transient_integral)

    S_lac_clean = (1.0 - pars.vb) * L
    return Pe, L, S_lac_clean


def get_flip_schedules(cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.schedule.upper() == "VFA":
        return vfa_schedule(cfg.n_timepoints)
    return cfa_schedule(cfg.n_timepoints, cfg.cfa_pyr_deg, cfg.cfa_lac_deg)


def ratio_metric(
    S_lac: np.ndarray,
    S_pyr: np.ndarray,
    t: np.ndarray,
    mode: str = "auc_full",
    tmin: float = 0.0,
    tmax: float = None,
    tpoint: float = None,
) -> Tuple[float, float, float]:
    S_lac = np.maximum(np.asarray(S_lac, dtype=float), 0.0)
    S_pyr = np.maximum(np.asarray(S_pyr, dtype=float), 0.0)
    t = np.asarray(t, dtype=float)

    if mode == "auc_full":
        auc_l = trapz(S_lac, t)
        auc_p = trapz(S_pyr, t)
        return float(auc_l), float(auc_p), float(auc_l / auc_p if auc_p > 0 else np.nan)

    if mode == "auc_window":
        if tmax is None:
            tmax = float(t[-1])
        m = (t >= float(tmin)) & (t <= float(tmax))
        if np.count_nonzero(m) < 2:
            return np.nan, np.nan, np.nan
        auc_l = trapz(S_lac[m], t[m])
        auc_p = trapz(S_pyr[m], t[m])
        return float(auc_l), float(auc_p), float(auc_l / auc_p if auc_p > 0 else np.nan)

    if mode == "tpoint":
        if tpoint is None:
            tpoint = float(t[len(t) // 2])
        idx = int(np.argmin(np.abs(t - float(tpoint))))
        denom = S_pyr[idx]
        return np.nan, np.nan, float(S_lac[idx] / denom if denom > 0 else np.nan)

    raise ValueError(f"Unknown ratio mode: {mode}")


def simulate_one(
    cfg: SimConfig,
    pars: KineticParams,
    drv: DriverParams,
    rng: np.random.Generator,
    ratio_mode: str = "auc_full",
    ratio_tmin: float = 0.0,
    ratio_tmax: float = None,
    ratio_tpoint: float = None,
) -> Dict:
    t, S_pyr_clean = build_measured_pyruvate_driver(cfg, drv)
    theta_p, theta_l = get_flip_schedules(cfg)
    _, L_clean, S_lac_clean = integrate_measured_driver(
        t, S_pyr_clean, theta_p, theta_l, cfg.TR, pars
    )

    def add_noise(x: np.ndarray, snr: float) -> np.ndarray:
        sigma = max(float(np.max(np.abs(x))), 1e-8) / max(float(snr), 1e-3)
        return np.asarray(x, dtype=float) + rng.normal(0.0, sigma, size=np.asarray(x).shape)

    S_pyr = add_noise(S_pyr_clean, cfg.snr_pyr)
    S_lac = add_noise(S_lac_clean, cfg.snr_lac)
    auc_l, auc_p, auc_ratio = ratio_metric(
        S_lac, S_pyr, t, mode=ratio_mode, tmin=ratio_tmin, tmax=ratio_tmax, tpoint=ratio_tpoint
    )

    return {
        "t": t,
        "theta_p": theta_p,
        "theta_l": theta_l,
        "S_p": S_pyr,
        "S_l": S_lac,
        "S_p_clean": S_pyr_clean,
        "S_l_clean": S_lac_clean,
        "L_clean": L_clean,
        "auc_l": float(auc_l),
        "auc_p": float(auc_p),
        "auc_ratio": float(auc_ratio),
        "aif_mode": drv.aif_mode,
        "t_shift": float(drv.t_shift),
        "aif_alpha": float(drv.aif_alpha),
        "aif_beta": float(drv.aif_beta),
        "aif_t0": float(drv.aif_t0),
        "aif_tau": float(drv.aif_tau),
    }


# -----------------------------------------------------------------------------
# Estimators
# -----------------------------------------------------------------------------


def fit_measured_driver_lactate(
    t: np.ndarray,
    S_pyr_obs: np.ndarray,
    S_lac_obs: np.ndarray,
    theta_p: np.ndarray,
    theta_l: np.ndarray,
    cfg: SimConfig,
    init: Tuple[float, float, float],
    bounds: Tuple[List[float], List[float]],
) -> Dict[str, float]:
    """NLLS fit using observed S_pyr as the measured driver."""

    def model_lac(_t, kpl, kve, vb):
        pars = KineticParams(kpl=float(kpl), kve=float(kve), vb=float(vb), r1p=cfg.r1p, r1l=cfg.r1l)
        _, _, S_lac_pred = integrate_measured_driver(t, S_pyr_obs, theta_p, theta_l, cfg.TR, pars)
        return S_lac_pred

    try:
        popt, _ = curve_fit(model_lac, t, S_lac_obs, p0=init, bounds=bounds, maxfev=10000)
        return {"kpl_hat": float(popt[0]), "kve_hat": float(popt[1]), "vb_hat": float(popt[2])}
    except Exception:
        return {"kpl_hat": np.nan, "kve_hat": np.nan, "vb_hat": np.nan}


class NNPredictor:
    def __init__(self, args: argparse.Namespace, cfg: SimConfig):
        import torch
        import torch.nn as nn

        class HybridMultiHead(nn.Module):
            def __init__(self, input_dim_raw, input_dim_norm, vb_range, kpl_range, kve_range):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(input_dim_norm, 128), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1)
                )
                self.kpl_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
                self.kve_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
                self.vb_head = nn.Sequential(
                    nn.Linear(64 + input_dim_raw, 64), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(64, 1), nn.Sigmoid()
                )
                vb_min, vb_max = vb_range
                kpl_min, kpl_max = kpl_range
                kve_min, kve_max = kve_range
                self.register_buffer("vb_min", torch.tensor([vb_min], dtype=torch.float32))
                self.register_buffer("vb_span", torch.tensor([vb_max - vb_min], dtype=torch.float32))
                self.register_buffer("kpl_min", torch.tensor([kpl_min], dtype=torch.float32))
                self.register_buffer("kpl_span", torch.tensor([kpl_max - kpl_min], dtype=torch.float32))
                self.register_buffer("kve_min", torch.tensor([kve_min], dtype=torch.float32))
                self.register_buffer("kve_span", torch.tensor([kve_max - kve_min], dtype=torch.float32))

            def forward(self, x_norm, x_raw):
                shared_out = self.shared(x_norm)
                kpl_hat = self.kpl_head(shared_out)
                kve_hat = self.kve_head(shared_out)
                vb_hat = self.vb_head(torch.cat([shared_out, x_raw], dim=1))
                kpl = self.kpl_min + kpl_hat * self.kpl_span
                kve = self.kve_min + kve_hat * self.kve_span
                vb = self.vb_min + vb_hat * self.vb_span
                return torch.cat([kpl, kve, vb], dim=1)

        self.torch = torch
        self.args = args
        self.cfg = cfg
        self.flatten_order = args.nn_flatten_order

        vb_range = parse_range(args.vb_range, (0.005, 0.20))
        kpl_range = parse_range(args.kpl_range, (0.001, 0.20))
        kve_range = parse_range(args.kve_range, (0.0, 0.50))

        self.model = HybridMultiHead(
            input_dim_raw=cfg.n_timepoints * 2,
            input_dim_norm=cfg.n_timepoints * 2,
            vb_range=vb_range,
            kpl_range=kpl_range,
            kve_range=kve_range,
        )

        state = torch.load(args.nn_weights, map_location=torch.device("cpu"))
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

    def _prepare_time_channel(self, S_p: np.ndarray, S_l: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from hybrid_model_utils import prepare_hybrid_inputs
            X = np.stack([S_p, S_l], axis=-1)[None, :, :].astype(np.float32)  # (1,T,2)
            pyr_peak = max(float(np.max(np.abs(S_p))), 1e-12)
            alpha = (float(self.args.nn_ptrain) / pyr_peak) if self.args.nn_use_robust_peak else 1.0
            X_norm, X_raw, _ = prepare_hybrid_inputs(X, alpha=alpha, flatten=True)
            return X_norm.astype(np.float32), X_raw.astype(np.float32)
        except Exception:
            # Explicit fallback equivalent for one sample.
            pyr_peak = max(float(np.max(np.abs(S_p))), 1e-12)
            alpha = (float(self.args.nn_ptrain) / pyr_peak) if self.args.nn_use_robust_peak else 1.0
            raw_tc = np.stack([S_p * alpha, S_l * alpha], axis=-1)[None, :, :].astype(np.float32)
            norm_tc = raw_tc / max(float(np.max(raw_tc[0, :, 0])), 1e-12)
            return norm_tc.reshape(1, -1), raw_tc.reshape(1, -1)

    def _prepare_channel_time(self, S_p: np.ndarray, S_l: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pyr_peak = max(float(np.max(np.abs(S_p))), 1e-12)
        alpha = (float(self.args.nn_ptrain) / pyr_peak) if self.args.nn_use_robust_peak else 1.0
        S_p_raw = S_p * alpha
        S_l_raw = S_l * alpha
        raw = np.concatenate([S_p_raw, S_l_raw]).astype(np.float32)[None, :]
        norm = np.concatenate([S_p_raw / max(np.max(S_p_raw), 1e-12), S_l_raw / max(np.max(S_p_raw), 1e-12)]).astype(np.float32)[None, :]
        return norm, raw

    def predict(self, S_p: np.ndarray, S_l: np.ndarray) -> Dict[str, float]:
        if self.flatten_order == "channel_time":
            X_norm, X_raw = self._prepare_channel_time(S_p, S_l)
        else:
            X_norm, X_raw = self._prepare_time_channel(S_p, S_l)

        with self.torch.no_grad():
            y = self.model(self.torch.from_numpy(X_norm), self.torch.from_numpy(X_raw)).cpu().numpy()[0]
        return {"kpl_hat": float(y[0]), "kve_hat": float(y[1]), "vb_hat": float(y[2])}


def estimate_params(sim: Dict, cfg: SimConfig, args: argparse.Namespace, nn_pred: NNPredictor, init: Tuple[float, float, float]) -> Dict[str, float]:
    if args.estimate == "none":
        return {"kpl_hat": np.nan, "kve_hat": np.nan, "vb_hat": np.nan}

    if args.estimate in {"measured_driver", "nlls"}:
        return fit_measured_driver_lactate(
            sim["t"],
            sim["S_p"],
            sim["S_l"],
            sim["theta_p"],
            sim["theta_l"],
            cfg,
            init=init,
            bounds=([args.kpl_min_fit, args.kve_min_fit, args.vb_min_fit], [args.kpl_max_fit, args.kve_max_fit, args.vb_max_fit]),
        )

    if args.estimate == "nn":
        if nn_pred is None:
            raise RuntimeError("NN predictor was not initialized.")
        return nn_pred.predict(sim["S_p"], sim["S_l"])

    raise ValueError(f"Unknown estimator: {args.estimate}")


# -----------------------------------------------------------------------------
# Matched-ratio experiment
# -----------------------------------------------------------------------------


def choose_driver_params(args: argparse.Namespace, rng: np.random.Generator, aif_mode: str, resample: set) -> DriverParams:
    alpha_list = parse_float_list(args.aif_alpha_list, [1.5, 2.0, 2.5, 3.0])
    beta_list = parse_float_list(args.aif_beta_list, [4.0, 6.0, 8.0])
    t0_list = parse_float_list(args.aif_t0_list, [1.0, 2.0, 3.0])
    tau_list = parse_float_list(args.aif_tau_list, [10.0, 20.0, 30.0])
    tshift_list = parse_float_list(args.t_shift_list, [-10.0, -5.0, 0.0, 5.0, 10.0])

    use_shape = "aifshape" in resample
    return DriverParams(
        aif_mode=aif_mode,
        t_shift=float(rng.choice(tshift_list)) if "tshift" in resample else float(args.t_shift),
        aif_alpha=float(rng.choice(alpha_list)) if (use_shape and aif_mode == "bolus") else float(args.aif_alpha),
        aif_beta=float(rng.choice(beta_list)) if (use_shape and aif_mode == "bolus") else float(args.aif_beta),
        aif_t0=float(rng.choice(t0_list)) if use_shape else float(args.aif_t0),
        aif_tau=float(rng.choice(tau_list)) if (use_shape and aif_mode == "infusion") else float(args.aif_tau),
    )


def choose_kinetic_params(args: argparse.Namespace, cfg: SimConfig, rng: np.random.Generator, kpl: float, resample: set) -> KineticParams:
    vb_list = parse_float_list(args.vb_list, [0.01, 0.02, 0.04, 0.06, 0.08])
    kve_list = parse_float_list(args.kve_list, [0.10, 0.15, 0.20, 0.25, 0.35])
    vb = float(rng.choice(vb_list)) if "vb" in resample else float(args.vb)
    kve = float(rng.choice(kve_list)) if "kve" in resample else float(args.kve)
    return KineticParams(kpl=float(kpl), kve=kve, vb=vb, r1p=cfg.r1p, r1l=cfg.r1l)


def row_from_sample(
    sim: Dict,
    est: Dict[str, float],
    pars: KineticParams,
    y: int,
    match_id: int,
    args: argparse.Namespace,
    tries: int,
) -> Dict:
    return {
        "experiment": "matchratio_corrected",
        "y": int(y),
        "match_id": int(match_id),
        "tries_for_positive_match": int(tries),
        "aif_mode": sim["aif_mode"],
        "t_shift": sim["t_shift"],
        "aif_alpha": sim["aif_alpha"],
        "aif_beta": sim["aif_beta"],
        "aif_t0": sim["aif_t0"],
        "aif_tau": sim["aif_tau"],
        "kpl_true": float(pars.kpl),
        "kve_true": float(pars.kve),
        "vb_true": float(pars.vb),
        "r1p_true": float(pars.r1p),
        "r1l_true": float(pars.r1l),
        "snr_pyr": float(args.snr_pyr),
        "snr_lac": float(args.snr_lac),
        "gamma": 1.0,  # retained only to make explicit that no metabolite-specific gain was used
        "auc_l": sim["auc_l"],
        "auc_p": sim["auc_p"],
        "auc_ratio": sim["auc_ratio"],
        "kpl_hat": float(est.get("kpl_hat", np.nan)),
        "kve_hat": float(est.get("kve_hat", np.nan)),
        "vb_hat": float(est.get("vb_hat", np.nan)),
    }


def run_matchratio(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict]:
    rng = set_seed(args.seed)
    cfg = SimConfig(
        TR=args.TR,
        n_timepoints=args.n_timepoints,
        schedule=args.schedule,
        cfa_pyr_deg=args.cfa_pyr_deg,
        cfa_lac_deg=args.cfa_lac_deg,
        snr_pyr=args.snr_pyr,
        snr_lac=args.snr_lac,
        r1p=args.r1p,
        r1l=args.r1l,
    )

    resample = {s.strip().lower() for s in args.match_resample.split(",") if s.strip()}
    if "gamma" in resample:
        raise ValueError(
            "Do not use gamma in the corrected Figure 5 matched-ratio experiment. "
            "Use --match_resample vb,kve,tshift,aifshape instead."
        )

    aif_modes = ["bolus", "infusion"] if args.aif_mode == "both" else [args.aif_mode]
    nn_pred = NNPredictor(args, cfg) if args.estimate == "nn" else None

    rows: List[Dict] = []
    attempted = 0
    accepted = 0
    tries_used = []

    for aif_mode in aif_modes:
        for match_id in range(int(args.n_pairs_requested)):
            attempted += 1

            # Negative class sample.
            pars_neg = choose_kinetic_params(args, cfg, rng, args.roc_kpl_neg, resample)
            drv_neg = choose_driver_params(args, rng, aif_mode, resample)
            sim_neg = simulate_one(
                cfg, pars_neg, drv_neg, rng,
                ratio_mode=args.ratio_mode,
                ratio_tmin=args.ratio_tmin,
                ratio_tmax=args.ratio_tmax,
                ratio_tpoint=args.ratio_tpoint,
            )
            target_ratio = float(sim_neg["auc_ratio"])
            if not np.isfinite(target_ratio):
                continue

            # Positive class sample: resample nuisance variables until AUC ratio matches.
            sim_pos = None
            pars_pos = None
            for n_try in range(1, int(args.match_max_tries) + 1):
                candidate_pars = choose_kinetic_params(args, cfg, rng, args.roc_kpl_pos, resample)
                candidate_drv = choose_driver_params(args, rng, aif_mode, resample)
                candidate = simulate_one(
                    cfg, candidate_pars, candidate_drv, rng,
                    ratio_mode=args.ratio_mode,
                    ratio_tmin=args.ratio_tmin,
                    ratio_tmax=args.ratio_tmax,
                    ratio_tpoint=args.ratio_tpoint,
                )
                cand_ratio = float(candidate["auc_ratio"])
                if np.isfinite(cand_ratio) and abs(cand_ratio - target_ratio) <= float(args.match_tol):
                    sim_pos = candidate
                    pars_pos = candidate_pars
                    break

            if sim_pos is None:
                continue

            accepted += 1
            tries_used.append(n_try)

            est_neg = estimate_params(sim_neg, cfg, args, nn_pred, init=(args.roc_kpl_neg, pars_neg.kve, pars_neg.vb))
            est_pos = estimate_params(sim_pos, cfg, args, nn_pred, init=(args.roc_kpl_pos, pars_pos.kve, pars_pos.vb))

            rows.append(row_from_sample(sim_neg, est_neg, pars_neg, 0, match_id, args, n_try))
            rows.append(row_from_sample(sim_pos, est_pos, pars_pos, 1, match_id, args, n_try))

    df = pd.DataFrame(rows)
    summary = summarize_results(df, args, attempted=attempted, accepted=accepted, tries_used=tries_used)
    return df, summary


# -----------------------------------------------------------------------------
# Summary and plotting
# -----------------------------------------------------------------------------


def auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m]
    s = s[m]
    if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    ranks = np.empty_like(s_sorted, dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = np.mean(np.arange(i + 1, j + 2))
        i = j + 1
    r_pos = ranks[y_sorted == 1]
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    return float((np.sum(r_pos) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve_manual(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    m = np.isfinite(y) & np.isfinite(s)
    y = y[m]
    s = s[m]
    if len(y) == 0 or np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    thresholds = np.r_[np.inf, np.sort(np.unique(s))[::-1], -np.inf]
    fpr, tpr = [], []
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    for th in thresholds:
        pred = s >= th
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        fpr.append(fp / n_neg)
        tpr.append(tp / n_pos)
    return np.asarray(fpr), np.asarray(tpr)


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    neg = df[df["y"] == 0].set_index("match_id")
    pos = df[df["y"] == 1].set_index("match_id")
    common = neg.index.intersection(pos.index)
    return pd.DataFrame({
        "match_id": common,
        "delta_auc_ratio": pos.loc[common, "auc_ratio"].to_numpy(float) - neg.loc[common, "auc_ratio"].to_numpy(float),
        "delta_kpl_hat": pos.loc[common, "kpl_hat"].to_numpy(float) - neg.loc[common, "kpl_hat"].to_numpy(float),
        "delta_kpl_true": pos.loc[common, "kpl_true"].to_numpy(float) - neg.loc[common, "kpl_true"].to_numpy(float),
    })


def summarize_results(df: pd.DataFrame, args: argparse.Namespace, attempted: int, accepted: int, tries_used: List[int]) -> Dict:
    if df.empty:
        return {
            "n_pairs": 0,
            "attempted_negative_samples": attempted,
            "accepted_pairs": accepted,
            "message": "No matched pairs accepted. Increase match_tol or match_max_tries, or broaden vB/kVE/timing/AIF-shape grids.",
        }

    y = df["y"].to_numpy(int)
    auc_ratio = df["auc_ratio"].to_numpy(float)
    kpl_hat = df["kpl_hat"].to_numpy(float)
    deltas = paired_deltas(df)

    summary = {
        "n_pairs": int(len(deltas)),
        "n_rows": int(len(df)),
        "attempted_negative_samples": int(attempted),
        "accepted_pairs": int(accepted),
        "pair_acceptance_fraction": float(accepted / max(attempted, 1)),
        "mean_tries_per_accepted_pair": float(np.mean(tries_used)) if tries_used else np.nan,
        "median_tries_per_accepted_pair": float(np.median(tries_used)) if tries_used else np.nan,
        "match_tolerance_abs_auc_ratio": float(args.match_tol),
        "resampled_confounds": args.match_resample,
        "gamma_used_for_matching": False,
        "auc_overall_LacPyr": auc_from_scores(y, auc_ratio),
        "auc_overall_kPLhat": auc_from_scores(y, kpl_hat),
        "median_abs_delta_auc_ratio": float(np.nanmedian(np.abs(deltas["delta_auc_ratio"]))) if not deltas.empty else np.nan,
        "mean_delta_auc_ratio": float(np.nanmean(deltas["delta_auc_ratio"])) if not deltas.empty else np.nan,
        "mean_delta_kPLhat": float(np.nanmean(deltas["delta_kpl_hat"])) if not deltas.empty else np.nan,
        "median_delta_kPLhat": float(np.nanmedian(deltas["delta_kpl_hat"])) if not deltas.empty else np.nan,
        "true_delta_kPL": float(args.roc_kpl_pos - args.roc_kpl_neg),
    }
    return summary



def estimator_label(args: argparse.Namespace) -> str:
    """Human-readable label for Figure 5 ROC legend and filenames."""
    if args.estimate == "nn":
        return "NN-estimated kPL"
    if args.estimate in {"measured_driver", "nlls"}:
        return "NLLS kPL"
    return "Estimated kPL"

def make_figure(df: pd.DataFrame, summary: Dict, args: argparse.Namespace) -> None:
    if not args.plot_out or df.empty:
        return
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    deltas = paired_deltas(df)
    y = df["y"].to_numpy(int)
    auc_ratio = df["auc_ratio"].to_numpy(float)
    kpl_hat = df["kpl_hat"].to_numpy(float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    fpr_a, tpr_a = roc_curve_manual(y, auc_ratio)
    fpr_k, tpr_k = roc_curve_manual(y, kpl_hat)
    axes[0].plot(fpr_a, tpr_a, label=f"Lac/Pyr AUC (AUC={summary['auc_overall_LacPyr']:.2f})")
    axes[0].plot(fpr_k, tpr_k, label=f"{estimator_label(args)} (AUC={summary['auc_overall_kPLhat']:.2f})")
    axes[0].plot([0, 1], [0, 1], "--", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("Matched-ratio discrimination")
    axes[0].legend(fontsize=8)

    if not deltas.empty:
        axes[1].hist(deltas["delta_auc_ratio"].dropna().to_numpy(float), bins=30)
        axes[1].axvline(0.0, linestyle="--", linewidth=1)
        axes[1].set_title("Paired Δ Lac/Pyr AUC")
        axes[1].set_xlabel("positive − negative")
        axes[1].set_ylabel("Matched pairs")
        axes[1].xaxis.set_major_locator(MaxNLocator(nbins=int(args.delta_auc_tick_bins)))
        axes[1].xaxis.set_major_formatter(FormatStrFormatter(f"%.{int(args.delta_auc_tick_decimals)}f"))

        axes[2].hist(deltas["delta_kpl_hat"].dropna().to_numpy(float), bins=30)
        axes[2].axvline(0.0, linestyle="--", linewidth=1)
        axes[2].axvline(args.roc_kpl_pos - args.roc_kpl_neg, linestyle=":", linewidth=1)
        axes[2].set_title("Paired Δ estimated kPL")
        axes[2].set_xlabel("positive − negative (s$^{-1}$)")
        axes[2].set_ylabel("Matched pairs")

    fig.suptitle(f"Figure 5 corrected: no pyruvate-only receive-gain term ({estimator_label(args)})", y=1.02)
    fig.tight_layout()
    fig.savefig(args.plot_out, dpi=int(args.plot_dpi), bbox_inches="tight")
    plt.close(fig)


def write_outputs(df: pd.DataFrame, summary: Dict, args: argparse.Namespace) -> None:
    if args.out:
        df.to_csv(args.out, index=False)
    if args.delta_out:
        paired_deltas(df).to_csv(args.delta_out, index=False)
    if args.summary_out:
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2)
    make_figure(df, summary, args)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Corrected Figure 5 matched-ratio HP-13C simulation")

    # outputs
    p.add_argument("--out", default="figure5_corrected_pairs.csv")
    p.add_argument("--delta_out", default="figure5_corrected_paired_deltas.csv")
    p.add_argument("--summary_out", default="figure5_corrected_summary.json")
    p.add_argument("--plot_out", default="figure5_corrected.png")
    p.add_argument("--plot_dpi", type=int, default=300)
    p.add_argument("--delta_auc_tick_decimals", type=int, default=3,
                   help="Decimal places for x-axis tick labels in paired Δ Lac/Pyr AUC panel.")
    p.add_argument("--delta_auc_tick_bins", type=int, default=5,
                   help="Maximum number of x-axis tick bins in paired Δ Lac/Pyr AUC panel.")

    # simulation size and matching
    p.add_argument("--n_pairs_requested", type=int, default=250)
    p.add_argument("--match_tol", type=float, default=0.01)
    p.add_argument("--match_max_tries", type=int, default=5000)
    p.add_argument("--match_resample", type=str, default="vb,kve,tshift,aifshape",
                   help="Allowed corrected confounds: vb,kve,tshift,aifshape. Do not include gamma.")
    p.add_argument("--seed", type=int, default=1337)

    # acquisition
    p.add_argument("--TR", type=float, default=2.0)
    p.add_argument("--n_timepoints", type=int, default=16)
    p.add_argument("--schedule", choices=["VFA", "CFA"], default="VFA")
    p.add_argument("--cfa_pyr_deg", type=float, default=11.0)
    p.add_argument("--cfa_lac_deg", type=float, default=80.0)
    p.add_argument("--snr_pyr", type=float, default=32.0)
    p.add_argument("--snr_lac", type=float, default=20.0)
    p.add_argument("--r1p", type=float, default=1.0 / 30.0)
    p.add_argument("--r1l", type=float, default=1.0 / 25.0)

    # kinetic truth and confound grids
    p.add_argument("--roc_kpl_neg", type=float, default=0.120)
    p.add_argument("--roc_kpl_pos", type=float, default=0.144)
    p.add_argument("--vb", type=float, default=0.02)
    p.add_argument("--vb_list", type=str, default="0.01,0.02,0.04,0.06,0.08")
    p.add_argument("--kve", type=float, default=0.25)
    p.add_argument("--kve_list", type=str, default="0.10,0.15,0.20,0.25,0.35")

    # driver/AIF timing and shape
    p.add_argument("--aif_mode", choices=["bolus", "infusion", "both"], default="bolus")
    p.add_argument("--t_shift", type=float, default=0.0)
    p.add_argument("--t_shift_list", type=str, default="-10,-5,0,5,10")
    p.add_argument("--aif_alpha", type=float, default=2.0)
    p.add_argument("--aif_beta", type=float, default=6.0)
    p.add_argument("--aif_t0", type=float, default=2.0)
    p.add_argument("--aif_tau", type=float, default=20.0)
    p.add_argument("--aif_alpha_list", type=str, default="1.5,2.0,2.5,3.0")
    p.add_argument("--aif_beta_list", type=str, default="4.0,6.0,8.0")
    p.add_argument("--aif_t0_list", type=str, default="1.0,2.0,3.0")
    p.add_argument("--aif_tau_list", type=str, default="10.0,20.0,30.0")

    # ratio definition
    p.add_argument("--ratio_mode", choices=["auc_full", "auc_window", "tpoint"], default="auc_full")
    p.add_argument("--ratio_tmin", type=float, default=0.0)
    p.add_argument("--ratio_tmax", type=float, default=None)
    p.add_argument("--ratio_tpoint", type=float, default=None)

    # estimator
    p.add_argument("--estimate", choices=["none", "measured_driver", "nlls", "nn"], default="none",
                   help="Estimator for kPL: 'nlls' is an alias for corrected measured-driver NLLS.")

    # measured-driver NLLS bounds
    p.add_argument("--kpl_min_fit", type=float, default=0.0)
    p.add_argument("--kpl_max_fit", type=float, default=0.5)
    p.add_argument("--kve_min_fit", type=float, default=0.0)
    p.add_argument("--kve_max_fit", type=float, default=1.0)
    p.add_argument("--vb_min_fit", type=float, default=0.005)
    p.add_argument("--vb_max_fit", type=float, default=0.20)

    # NN options
    p.add_argument("--nn_weights", type=str, default="")
    p.add_argument("--nn_ptrain", type=float, default=1.0)
    p.add_argument("--nn_use_robust_peak", action="store_true")
    p.add_argument("--nn_flatten_order", choices=["time_channel", "channel_time"], default="time_channel")
    p.add_argument("--kpl_range", type=str, default="0.001,0.200")
    p.add_argument("--kve_range", type=str, default="0.000,0.500")
    p.add_argument("--vb_range", type=str, default="0.005,0.200")

    args = p.parse_args()
    if args.estimate == "nn" and not args.nn_weights:
        raise ValueError("--nn_weights is required when --estimate nn")
    if args.estimate == "nn" and not os.path.exists(args.nn_weights):
        raise FileNotFoundError(f"NN weights not found: {args.nn_weights}")
    return args


def main() -> None:
    args = parse_args()
    df, summary = run_matchratio(args)
    write_outputs(df, summary, args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
