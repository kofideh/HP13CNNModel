#!/usr/bin/env python3
"""
SNR perturbation analysis for HP-13C pyruvate/lactate kinetic parameter recovery.

Primary gamma-AIF version.

1) Clean pyruvate/lactate curves are generated directly by the protocol-specific
   two-compartment physiologic generator, whose latent vascular input is a
   gamma-variate-like Pv(t). The lactate curve is not overwritten with a
   measured-driver equation.
2) Independent zero-mean Gaussian noise is added to pyruvate and lactate using
   the same rms sigma within each sample or condition. Metabolite-specific SNR
   differences therefore arise from signal amplitude differences, not different
   noise amplitudes.
3) NLLS uses fit_nlls_gamma_AIF(), which jointly fits kPL, kVE, vB, and the
   gamma-variate vascular-input parameters from the noisy pyruvate/lactate
   curves. This matches the primary in-vivo NLLS comparator.
4) The same clean parameter/time-course set is reused across SNR conditions by
   default so that the perturbation experiment isolates noise effects. Use
   --resample-each-snr for independent parameter draws at each SNR point.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib

from fit_two_compartment_physio_nlls import fit_nlls_gamma_AIF
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hybrid_model_utils import (
    HybridMultiHead,
    compute_peak_percentile,
    load_training_config,
    plot_prediction_scatter,
    prepare_hybrid_inputs,
)

PARAMETER_NAMES = ("kPL", "kVE", "vB")


def add_staggered_lactate_snr_axis(
    ax1: plt.Axes,
    snrs: np.ndarray,
    order: np.ndarray,
    xlabel: str = "Mean Lactate SNR",
    proximity_fraction: float = 0.045,
) -> plt.Axes:
    """Add a top x-axis with lactate SNR labels without overlapping close ticks.

    The bottom axis is mean pyruvate SNR. The top axis uses the same tick
    locations but labels each location by the corresponding mean lactate SNR.
    At low SNR, neighboring pyruvate-SNR tick locations can be very close,
    causing the lactate-SNR labels to overlap. We detect close neighbors and
    stagger the second label in each close pair onto a second line.
    """
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())

    x_ticks = np.asarray(snrs[order, 0], dtype=float)
    lactate_labels = [f"{v:.1f}" for v in np.asarray(snrs[order, 1], dtype=float)]

    xmin, xmax = ax1.get_xlim()
    axis_span = max(float(xmax - xmin), 1e-12)
    close_threshold = proximity_fraction * axis_span

    staggered = False
    for i in range(1, len(x_ticks)):
        if (x_ticks[i] - x_ticks[i - 1]) < close_threshold:
            lactate_labels[i] = "\n" + lactate_labels[i]
            staggered = True

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(lactate_labels)
    ax2.tick_params(axis="x", which="major", pad=2)
    ax2.set_xlabel(xlabel, labelpad=18 if staggered else 6)
    return ax2


def parse_float_list(text: str) -> list:
    return [float(v.strip()) for v in str(text).split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corrected SNR perturbation analysis for HP-13C kinetic model inference."
    )
    parser.add_argument(
        "--weights-dir",
        required=True,
        help="Subdirectory under output/ containing trained_hybrid_positive.pth and training_report.md.",
    )
    parser.add_argument("--output-dir", default="output", help="Root output directory. Default: output.")
    parser.add_argument("--n-samples", type=int, default=100, help="Samples per SNR condition. Default: 100.")
    parser.add_argument("--noise-seed", type=int, default=20260517, help="Seed for equal-rms noise. Default: 20260517.")
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Optional seed before generator calls if your generator uses NumPy global RNG.",
    )
    parser.add_argument(
        "--target-pyr-snrs",
        default="3,6,12,23,54,88",
        help=(
            "Comma-separated target pyruvate SNR values. With the default per-condition "
            "sigma mode, these are referenced to the selected clean pyruvate peak statistic. "
            "Default: 3,6,12,23,54,88."
        ),
    )
    parser.add_argument(
        "--sigma-mode",
        choices=("per-condition", "per-sample"),
        default="per-condition",
        help=(
            "per-condition: one rms sigma for all samples and both metabolites at each SNR setting; "
            "per-sample: one rms sigma per sample, still shared by pyruvate/lactate. Default: per-condition."
        ),
    )
    parser.add_argument(
        "--snr-reference",
        choices=("median", "mean", "max", "p90"),
        default="median",
        help=(
            "Clean pyruvate peak statistic used to convert target pyruvate SNR to condition sigma "
            "when --sigma-mode=per-condition. Default: median."
        ),
    )
    parser.add_argument(
        "--resample-each-snr",
        action="store_true",
        help="Generate a new clean dataset for each SNR point. Default is to reuse the same clean dataset.",
    )
    parser.add_argument(
        "--r1p",
        type=float,
        default=1.0 / 30.0,
        help="Generating/fitting pyruvate R1 in s^-1. Default: 1/30.",
    )
    parser.add_argument(
        "--r1l",
        type=float,
        default=1.0 / 25.0,
        help="Generating/fitting lactate R1 in s^-1. Default: 1/25.",
    )
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap iterations for R2 uncertainty. Default: 200.")
    parser.add_argument(
        "--use-weighted-nlls",
        action="store_true",
        help="Weight the pyruvate and lactate NLLS residuals by the known equal rms sigma.",
    )
    parser.add_argument(
        "--clip-negative-noisy-signal",
        action="store_true",
        help="Clip noisy signals at zero. Default is not to clip to preserve zero-mean Gaussian noise.",
    )
    args, _ = parser.parse_known_args()
    return args


def _as_schedule(x: Sequence[float], n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.repeat(float(arr), n)
    if arr.size == n:
        return arr.astype(float)
    if arr.size > n:
        return arr[:n].astype(float)
    return np.pad(arr.astype(float), (0, n - arr.size), mode="edge")


def build_generator(cfg: Dict[str, float], time_points: np.ndarray, R1p: float, R1l: float, seed: Optional[int] = None):
    # Build the same physiologic gamma-AIF generator used for synthetic NN training.
    from two_compartment_generator_physio import TwoCompartmentHPDataGeneratorPhysio
    return TwoCompartmentHPDataGeneratorPhysio(
        vb_range=(cfg["VB_MIN"], cfg["VB_MAX"]),
        kpl_range=(cfg["KPL_MIN"], cfg["KPL_MAX"]),
        kve_range=(cfg["KVE_MIN"], cfg["KVE_MAX"]),
        r1p_range=(float(R1p), float(R1p)),
        r1l_range=(float(R1l), float(R1l)),
        time_points=time_points,
        flip_angle_pyr_deg=cfg["PYR_FA_SCHEDULE"],
        flip_angle_lac_deg=cfg["LAC_FA_SCHEDULE"],
        TR=cfg["SCAN_TR"],
        seed=seed,
    )


def generate_dataset_noiseless(generator, n_samples: int):
    """Call project generator while suppressing generator-side noise as much as possible."""
    try:
        out = generator.generate_dataset(n_samples=n_samples, noise_std=0.0)
    except TypeError:
        try:
            out = generator.generate_dataset(n_samples=n_samples, snr_range=(1e12, 1e12))
        except TypeError:
            out = generator.generate_dataset(n_samples=n_samples)

    if isinstance(out, tuple):
        if len(out) == 4:
            X, y, met_pools, extra = out
        elif len(out) == 3:
            X, y, met_pools = out
            extra = None
        else:
            raise ValueError("Unexpected generator output tuple length.")
    else:
        raise ValueError("Generator must return a tuple containing X and y.")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), met_pools, extra


def _reference_peak(peak_pyr: np.ndarray, mode: str) -> float:
    if mode == "median":
        return float(np.nanmedian(peak_pyr))
    if mode == "mean":
        return float(np.nanmean(peak_pyr))
    if mode == "max":
        return float(np.nanmax(peak_pyr))
    if mode == "p90":
        return float(np.nanpercentile(peak_pyr, 90))
    raise ValueError(f"Unknown reference mode: {mode}")


def add_equal_rms_noise_for_snr(
    X_clean: np.ndarray,
    rng: np.random.Generator,
    target_pyr_snr: float,
    sigma_mode: str,
    snr_reference: str,
    clip_negative: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, float]:
    """Add equal-rms noise to both channels and return noisy data plus per-sample SNR audit table."""
    X_clean = np.asarray(X_clean, dtype=float)
    peak_pyr = np.maximum(np.nanmax(np.abs(X_clean[:, :, 0]), axis=1), 1e-12)
    peak_lac = np.nanmax(np.abs(X_clean[:, :, 1]), axis=1)
    n = X_clean.shape[0]
    target_pyr_snr = float(target_pyr_snr)

    if sigma_mode == "per-sample":
        sigma = peak_pyr / max(target_pyr_snr, 1e-12)
        sigma_condition = float(np.nanmedian(sigma))
    else:
        ref_peak = _reference_peak(peak_pyr, snr_reference)
        sigma_condition = ref_peak / max(target_pyr_snr, 1e-12)
        sigma = np.full(n, sigma_condition, dtype=float)

    noise = rng.normal(0.0, sigma[:, None, None], size=X_clean.shape)
    X_noisy = X_clean + noise
    if clip_negative:
        X_noisy = np.maximum(X_noisy, 0.0)

    noise_df = pd.DataFrame(
        {
            "sample": np.arange(n, dtype=int),
            "target_pyr_snr": target_pyr_snr,
            "sigma_equal_rms": sigma,
            "clean_peak_pyr": peak_pyr,
            "clean_peak_lac": peak_lac,
            "snr_pyr": peak_pyr / sigma,
            "snr_lac": peak_lac / sigma,
        }
    )
    return X_noisy, noise_df, sigma_condition


def prepare_inputs_for_model(X: np.ndarray, cfg: Dict[str, float]):
    P_clin = compute_peak_percentile(X, percentile=cfg["PERCENTILE"], pyr_channel=0, min_peak=1e-6)
    alpha = cfg["P_TRAIN"] / max(P_clin, 1e-8)
    X_norm, X_raw, clin_meta = prepare_hybrid_inputs(X, alpha=alpha, pyr_channel=0, flatten=True)
    Xr = X_raw.reshape(X.shape[0], -1)
    Xn = X_norm.reshape(X.shape[0], -1)
    return Xn, Xr, X_norm, X_raw, clin_meta, alpha, P_clin


def finite_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = np.full(y_true.shape[1], np.nan, dtype=float)
    for p_idx in range(y_true.shape[1]):
        mask = np.isfinite(y_true[:, p_idx]) & np.isfinite(y_pred[:, p_idx])
        if np.count_nonzero(mask) < 2:
            continue
        den = np.sum((y_true[mask, p_idx] - np.mean(y_true[mask, p_idx])) ** 2)
        if den <= 0:
            continue
        num = np.sum((y_pred[mask, p_idx] - y_true[mask, p_idx]) ** 2)
        out[p_idx] = 1.0 - num / den
    return out


def bootstrap_r2_std(y_true: np.ndarray, y_pred: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    n, p = y_true.shape
    vals = np.full((max(B, 1), p), np.nan, dtype=float)
    if n < 2 or B <= 1:
        return np.full(p, np.nan, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        vals[b, :] = finite_r2(y_true[idx], y_pred[idx])
    return np.nanstd(vals, axis=0, ddof=1)


def finite_cov(values: np.ndarray) -> np.ndarray:
    """Coefficient of variation, std/|mean|, computed columnwise with NaN handling."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    out = np.full(values.shape[1], np.nan, dtype=float)
    eps = 1e-12
    for p_idx in range(values.shape[1]):
        v = values[:, p_idx]
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        mu = float(np.mean(v))
        sd = float(np.std(v, ddof=1))
        if abs(mu) > eps:
            out[p_idx] = sd / abs(mu)
    return out


def finite_error_cov(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Residual CoV: std(pred-true)/|mean(true)|, columnwise."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = np.full(y_true.shape[1], np.nan, dtype=float)
    eps = 1e-12
    for p_idx in range(y_true.shape[1]):
        mask = np.isfinite(y_true[:, p_idx]) & np.isfinite(y_pred[:, p_idx])
        if np.count_nonzero(mask) < 2:
            continue
        denom = float(np.mean(y_true[mask, p_idx]))
        resid_sd = float(np.std(y_pred[mask, p_idx] - y_true[mask, p_idx], ddof=1))
        if abs(denom) > eps:
            out[p_idx] = resid_sd / abs(denom)
    return out


def save_true_pred_csv(run_dir: str, basename: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    df = pd.DataFrame(
        {
            "kPL_true": y_true[:, 0],
            "kVE_true": y_true[:, 1],
            "vB_true": y_true[:, 2],
            "kPL_pred": y_pred[:, 0],
            "kVE_pred": y_pred[:, 1],
            "vB_pred": y_pred[:, 2],
        }
    )
    path = os.path.join(run_dir, f"{basename}.csv")
    df.to_csv(path, index=False)
    return path


def run_nlls_for_condition(
    X: np.ndarray,
    noise_df: pd.DataFrame,
    cfg: Dict[str, float],
    time_points: np.ndarray,
    R1p: float,
    R1l: float,
    use_weighted_nlls: bool,
) -> Tuple[np.ndarray, int, float]:
    bounds = (
        (cfg["KPL_MIN"], cfg["KVE_MIN"], cfg["VB_MIN"]),
        (cfg["KPL_MAX"], cfg["KVE_MAX"], cfg["VB_MAX"]),
    )
    p0 = (
        0.5 * (cfg["KPL_MIN"] + cfg["KPL_MAX"]),
        0.5 * (cfg["KVE_MIN"] + cfg["KVE_MAX"]),
        0.5 * (cfg["VB_MIN"] + cfg["VB_MAX"]),
    )

    preds = []
    failures = 0
    start = time.time()
    for j in range(X.shape[0]):
        try:
            sigma = float(noise_df.loc[j, "sigma_equal_rms"])
            res = fit_nlls_gamma_AIF(
                time_points=time_points,
                S_pyr_obs=X[j, :, 0],
                S_lac_obs=X[j, :, 1],
                TR=cfg["SCAN_TR"],
                flips_pyr_deg=cfg["PYR_FA_SCHEDULE"],
                flips_lac_deg=cfg["LAC_FA_SCHEDULE"],
                R1p=R1p,
                R1l=R1l,
                p0=p0,
                bounds=bounds,
                sigma_pyr=sigma if use_weighted_nlls else None,
                sigma_lac=sigma if use_weighted_nlls else None,
            )
            if res.get("success", False):
                preds.append([res["kpl"], res["kve"], res["vb"]])
            else:
                preds.append([np.nan, np.nan, np.nan])
                failures += 1
        except Exception:
            preds.append([np.nan, np.nan, np.nan])
            failures += 1
    return np.asarray(preds, dtype=float), failures, time.time() - start


def plot_snr_r2_summary(
    run_dir: str,
    snrs: np.ndarray,
    r2_nn: np.ndarray,
    r2_nlls: np.ndarray,
    r2_nn_std: np.ndarray,
    r2_nlls_std: np.ndarray,
) -> str:
    fig, ax1 = plt.subplots(figsize=(8.5, 6.0))
    colors = {"kPL": "C0", "kVE": "C1", "vB": "C2"}
    x = snrs[:, 0]
    order = np.argsort(x)
    for p_idx, pname in enumerate(PARAMETER_NAMES):
        ax1.plot(x[order], r2_nn[order, p_idx], label=f"NN R$^2$ {pname}", marker="o", color=colors[pname], linestyle="-")
        ax1.plot(x[order], r2_nlls[order, p_idx], label=f"NLLS R$^2$ {pname}", marker="x", color=colors[pname], linestyle="--")
        if np.any(np.isfinite(r2_nn_std[:, p_idx])):
            ax1.fill_between(x[order], (r2_nn[:, p_idx] - r2_nn_std[:, p_idx])[order], (r2_nn[:, p_idx] + r2_nn_std[:, p_idx])[order], color=colors[pname], alpha=0.15)
        if np.any(np.isfinite(r2_nlls_std[:, p_idx])):
            ax1.fill_between(x[order], (r2_nlls[:, p_idx] - r2_nlls_std[:, p_idx])[order], (r2_nlls[:, p_idx] + r2_nlls_std[:, p_idx])[order], color=colors[pname], alpha=0.08)

    ax1.set_xlabel("Mean Pyruvate SNR")
    ax1.set_ylabel("Coefficient of Determination (R$^2$)")
    ax1.set_title("SNR perturbation: parameter recovery accuracy")
    ax1.set_ylim(-0.5, 1.05)
    ax1.grid(True, which="major", alpha=0.7)
    ax1.grid(True, which="minor", alpha=0.3, linestyle=":")

    add_staggered_lactate_snr_axis(ax1, snrs, order)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    out = os.path.join(run_dir, "SNR_vs_R2_all_params_gammaAIF.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_snr_cov_summary(
    run_dir: str,
    snrs: np.ndarray,
    cov_nn: np.ndarray,
    cov_nlls: np.ndarray,
    cov_error_nn: Optional[np.ndarray] = None,
    cov_error_nlls: Optional[np.ndarray] = None,
) -> Tuple[str, Optional[str]]:
    colors = {"kPL": "C0", "kVE": "C1", "vB": "C2"}
    x = snrs[:, 0]
    order = np.argsort(x)

    def _plot_one(arr_nn: np.ndarray, arr_nlls: np.ndarray, ylabel: str, title: str, fname: str) -> str:
        fig, ax1 = plt.subplots(figsize=(8.5, 6.0))
        for p_idx, pname in enumerate(PARAMETER_NAMES):
            ax1.plot(x[order], arr_nn[order, p_idx], label=f"NN CoV {pname}", marker="o", color=colors[pname], linestyle="-")
            ax1.plot(x[order], arr_nlls[order, p_idx], label=f"NLLS CoV {pname}", marker="x", color=colors[pname], linestyle="--")
        ax1.set_xlabel("Mean Pyruvate SNR")
        ax1.set_ylabel(ylabel)
        ax1.set_title(title)
        ax1.set_ylim(bottom=0)
        ax1.grid(True, which="major", alpha=0.7)
        ax1.grid(True, which="minor", alpha=0.3, linestyle=":")

        add_staggered_lactate_snr_axis(ax1, snrs, order)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10)
        fig.tight_layout(rect=[0, 0.12, 1, 1])
        out = os.path.join(run_dir, fname)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    pred_cov_path = _plot_one(
        cov_nn,
        cov_nlls,
        ylabel="Coefficient of variation of predicted parameters",
        title="SNR perturbation: prediction coefficient of variation",
        fname="SNR_vs_CoV_predicted_params_gammaAIF.png",
    )

    err_cov_path = None
    if cov_error_nn is not None and cov_error_nlls is not None:
        err_cov_path = _plot_one(
            cov_error_nn,
            cov_error_nlls,
            ylabel="Residual CoV: std(pred−true)/|mean(true)|",
            title="SNR perturbation: residual coefficient of variation",
            fname="SNR_vs_CoV_residual_error_gammaAIF.png",
        )
    return pred_cov_path, err_cov_path


def main() -> None:
    args = parse_args()

    weights_path = os.path.join(args.output_dir, args.weights_dir, "trained_hybrid_positive.pth")
    training_data_info_path = os.path.join(args.output_dir, args.weights_dir, "training_report.md")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")
    if not os.path.exists(training_data_info_path):
        raise FileNotFoundError(f"Training report not found: {training_data_info_path}")

    cfg = load_training_config(training_data_info_path)
    target_pyr_snrs = parse_float_list(args.target_pyr_snrs)
    if len(target_pyr_snrs) == 0:
        raise ValueError("At least one target pyruvate SNR must be supplied.")

    n_timepoints = int(cfg["NUM_TIMEPOINTS"])
    time_points = np.arange(0, n_timepoints * float(cfg["SCAN_TR"]), float(cfg["SCAN_TR"]))

    model = HybridMultiHead(
        input_dim_raw=n_timepoints * 2,
        input_dim_norm=n_timepoints * 2,
        vb_range=(cfg["VB_MIN"], cfg["VB_MAX"]),
        kpl_range=(cfg["KPL_MIN"], cfg["KPL_MAX"]),
        kve_range=(cfg["KVE_MIN"], cfg["KVE_MAX"]),
    )
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model.eval()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output_dir, args.weights_dir, f"SNRvariation_gammaAIF_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    n_cond = len(target_pyr_snrs)
    snrs = np.full((n_cond, 2), np.nan)
    snrs_median = np.full((n_cond, 2), np.nan)
    sigma_condition_arr = np.full(n_cond, np.nan)
    r2_nn = np.full((n_cond, 3), np.nan)
    r2_nlls = np.full((n_cond, 3), np.nan)
    r2_nn_std = np.full((n_cond, 3), np.nan)
    r2_nlls_std = np.full((n_cond, 3), np.nan)
    cov_nn = np.full((n_cond, 3), np.nan)
    cov_nlls = np.full((n_cond, 3), np.nan)
    cov_error_nn = np.full((n_cond, 3), np.nan)
    cov_error_nlls = np.full((n_cond, 3), np.nan)
    nlls_failures_arr = np.zeros(n_cond, dtype=int)
    nlls_time_arr = np.zeros(n_cond, dtype=float)

    rng_noise = np.random.default_rng(args.noise_seed)
    rng_boot = np.random.default_rng(args.noise_seed + 1)

    generator = build_generator(cfg, time_points, args.r1p, args.r1l, seed=args.data_seed)
    X_clean_base = None
    y_true_base = None
    met_pools_base = None

    if not args.resample_each_snr:
        if args.data_seed is not None:
            np.random.seed(int(args.data_seed))
        X_clean_base, y_true_base, met_pools_base, _ = generate_dataset_noiseless(
            generator=generator,
            n_samples=args.n_samples,
        )

    all_summary_rows = []

    for cond_idx, target_snr in enumerate(target_pyr_snrs):
        print(f"\n=== SNR condition {cond_idx+1}/{n_cond}: target pyruvate SNR={target_snr:g} ===")

        if args.resample_each_snr:
            if args.data_seed is not None:
                generator_cond = build_generator(cfg, time_points, args.r1p, args.r1l, seed=int(args.data_seed) + cond_idx)
            else:
                generator_cond = generator
            X_clean, y_true, met_pools, _ = generate_dataset_noiseless(
                generator=generator_cond,
                n_samples=args.n_samples,
            )
        else:
            X_clean = X_clean_base.copy()
            y_true = y_true_base.copy()
            met_pools = met_pools_base

        X, noise_df, sigma_condition = add_equal_rms_noise_for_snr(
            X_clean=X_clean,
            rng=rng_noise,
            target_pyr_snr=target_snr,
            sigma_mode=args.sigma_mode,
            snr_reference=args.snr_reference,
            clip_negative=args.clip_negative_noisy_signal,
        )
        label = f"targetPyrSNR_{target_snr:g}".replace(".", "p")
        noise_df.to_csv(os.path.join(run_dir, f"per_sample_equal_rms_noise_{label}.csv"), index=False)

        snrs[cond_idx, 0] = float(noise_df["snr_pyr"].mean())
        snrs[cond_idx, 1] = float(noise_df["snr_lac"].mean())
        snrs_median[cond_idx, 0] = float(noise_df["snr_pyr"].median())
        snrs_median[cond_idx, 1] = float(noise_df["snr_lac"].median())
        sigma_condition_arr[cond_idx] = sigma_condition

        Xn, Xr, X_norm, X_raw, clin_meta, alpha, P_clin = prepare_inputs_for_model(X, cfg)
        with torch.no_grad():
            y_pred_nn = model(torch.tensor(Xn, dtype=torch.float32), torch.tensor(Xr, dtype=torch.float32)).cpu().numpy()

        y_pred_nlls, failures, elapsed = run_nlls_for_condition(
            X=X,
            noise_df=noise_df,
            cfg=cfg,
            time_points=time_points,
            R1p=args.r1p,
            R1l=args.r1l,
            use_weighted_nlls=args.use_weighted_nlls,
        )
        nlls_failures_arr[cond_idx] = failures
        nlls_time_arr[cond_idx] = elapsed
        print(
            f"Mean SNR Pyr/Lac={snrs[cond_idx,0]:.2f}/{snrs[cond_idx,1]:.2f}; "
            f"NLLS failures: {failures}/{X.shape[0]}; elapsed={elapsed:.2f}s"
        )

        r2_nn[cond_idx, :] = finite_r2(y_true[:, :3], y_pred_nn[:, :3])
        r2_nlls[cond_idx, :] = finite_r2(y_true[:, :3], y_pred_nlls[:, :3])
        r2_nn_std[cond_idx, :] = bootstrap_r2_std(y_true[:, :3], y_pred_nn[:, :3], args.bootstrap, rng_boot)
        r2_nlls_std[cond_idx, :] = bootstrap_r2_std(y_true[:, :3], y_pred_nlls[:, :3], args.bootstrap, rng_boot)

        cov_nn[cond_idx, :] = finite_cov(y_pred_nn[:, :3])
        cov_nlls[cond_idx, :] = finite_cov(y_pred_nlls[:, :3])
        cov_error_nn[cond_idx, :] = finite_error_cov(y_true[:, :3], y_pred_nn[:, :3])
        cov_error_nlls[cond_idx, :] = finite_error_cov(y_true[:, :3], y_pred_nlls[:, :3])

        save_true_pred_csv(run_dir, f"TrueVPred_{label}_NN", y_true[:, :3], y_pred_nn[:, :3])
        save_true_pred_csv(run_dir, f"TrueVPred_{label}_NLLS", y_true[:, :3], y_pred_nlls[:, :3])
        try:
            plot_prediction_scatter(y_true[:, :3], y_pred_nn[:, :3], saveName=os.path.join(run_dir, f"NeuralNetwork_{label}"))
            plot_prediction_scatter(y_true[:, :3], y_pred_nlls[:, :3], saveName=os.path.join(run_dir, f"NLLS_{label}"))
        except Exception as exc:
            print(f"Warning: plot_prediction_scatter2 failed for {label}: {exc}")

        row = {
            "target_SNR_Pyruvate": target_snr,
            "mean_SNR_Pyruvate": snrs[cond_idx, 0],
            "mean_SNR_Lactate": snrs[cond_idx, 1],
            "median_SNR_Pyruvate": snrs_median[cond_idx, 0],
            "median_SNR_Lactate": snrs_median[cond_idx, 1],
            "sigma_condition_or_median": sigma_condition,
            "sigma_mode": args.sigma_mode,
            "snr_reference": args.snr_reference,
            "nlls_failures": failures,
            "nlls_elapsed_s": elapsed,
            "alpha_input_alignment": alpha,
            "P_clin": P_clin,
        }
        for p_idx, pname in enumerate(PARAMETER_NAMES):
            row[f"R2_NN_{pname}"] = r2_nn[cond_idx, p_idx]
            row[f"R2_NN_{pname}_std"] = r2_nn_std[cond_idx, p_idx]
            row[f"R2_NLLS_{pname}"] = r2_nlls[cond_idx, p_idx]
            row[f"R2_NLLS_{pname}_std"] = r2_nlls_std[cond_idx, p_idx]
            row[f"CoV_pred_NN_{pname}"] = cov_nn[cond_idx, p_idx]
            row[f"CoV_pred_NLLS_{pname}"] = cov_nlls[cond_idx, p_idx]
            row[f"CoV_residual_NN_{pname}"] = cov_error_nn[cond_idx, p_idx]
            row[f"CoV_residual_NLLS_{pname}"] = cov_error_nlls[cond_idx, p_idx]
        all_summary_rows.append(row)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_csv = os.path.join(run_dir, "SNR_R2_summary_gammaAIF.csv")
    summary_xlsx = os.path.join(run_dir, "SNR_R2_summary_gammaAIF.xlsx")
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_excel(summary_xlsx, index=False)

    fig_path = plot_snr_r2_summary(run_dir, snrs, r2_nn, r2_nlls, r2_nn_std, r2_nlls_std)
    cov_fig_path, cov_err_fig_path = plot_snr_cov_summary(
        run_dir, snrs, cov_nn, cov_nlls, cov_error_nn, cov_error_nlls
    )

    with open(os.path.join(run_dir, "summary_report.md"), "w", encoding="utf-8") as f:
        f.write("# Gamma-AIF SNR perturbation analysis\n\n")
        f.write(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"Samples per SNR setting: {args.n_samples}\n")
        f.write(f"Target pyruvate SNRs: {target_pyr_snrs}\n")
        f.write(f"Sigma mode: {args.sigma_mode}\n")
        f.write(f"SNR reference statistic: {args.snr_reference}\n")
        f.write(f"R1p/R1l used for simulation and NLLS: {args.r1p:g}/{args.r1l:g} s^-1\n\n")
        f.write("## Primary analysis choices\n")
        f.write("- Clean pyruvate/lactate curves were generated directly by the gamma-variate vascular-input two-compartment generator.\n")
        f.write("- The lactate channel was not overwritten with a measured-driver equation.\n")
        f.write("- Independent Gaussian noise was added to pyruvate and lactate using the same rms sigma per sample/condition.\n")
        f.write("- Pyruvate/lactate SNRs were computed as clean peak signal divided by the same sigma.\n")
        f.write("- NLLS used fit_nlls_gamma_AIF(), jointly fitting kPL, kVE, vB, and gamma-variate vascular-input parameters.\n\n")
        f.write(f"Summary CSV: {os.path.basename(summary_csv)}\n")
        f.write(f"Summary figure: {os.path.basename(fig_path)}\n")
        f.write(f"Prediction CoV figure: {os.path.basename(cov_fig_path)}\n")
        if cov_err_fig_path is not None:
            f.write(f"Residual CoV figure: {os.path.basename(cov_err_fig_path)}\n")

    print("\n" + "=" * 80)
    print("GAMMA-AIF SNR PERTURBATION ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {run_dir}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary figure: {fig_path}")
    print(f"Prediction CoV figure: {cov_fig_path}")
    if cov_err_fig_path is not None:
        print(f"Residual CoV figure: {cov_err_fig_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
