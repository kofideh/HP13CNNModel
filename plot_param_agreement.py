#!/usr/bin/env python3
import argparse, os, numpy as np, matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import r2_score
from scipy.stats import norm

# ---------- metrics ----------
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def deming(x, y, lam=1.0):
    x = np.asarray(x); y = np.asarray(y)
    xbar, ybar = np.mean(x), np.mean(y)
    Sxx = np.var(x, ddof=1); Syy = np.var(y, ddof=1); Sxy = np.cov(x, y, ddof=1)[0,1]
    B = Syy - lam*Sxx
    slope = (B + np.sqrt(B**2 + 4*lam*Sxy**2)) / (2*Sxy + 1e-12)
    intercept = ybar - slope*xbar
    return float(slope), float(intercept)

def deming_ci(x, y, lam=1.0, alpha=0.05, B=2000):
    slopes = []
    for _ in range(B):
        xb, yb = resample(x, y)
        slopes.append(deming(xb, yb, lam=lam)[0])
    slopes = np.sort(slopes)
    lo = slopes[int((alpha/2)*B)]
    hi = slopes[int((1-alpha/2)*B)-1]
    return float(lo), float(hi)

def bootstrap_r2_score(x_true, y_pred, B=200):
    vals = []
    n = len(x_true)
    x_true = np.asarray(x_true); y_pred = np.asarray(y_pred)
    for _ in range(B):
        idx = np.random.randint(0, n, n)
        vals.append(r2_score(x_true[idx], y_pred[idx]))
    return float(np.mean(vals)), float(np.std(vals, ddof=1))

def ccc(x_true, y_pred):
    """Lin's Concordance Correlation Coefficient."""
    x = np.asarray(x_true); y = np.asarray(y_pred)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0,1]
    denom = vx + vy + (mx - my)**2
    if denom <= 0: 
        return np.nan
    return float(2*sxy / denom)

def bland_altman_stats(x_true, y_pred):
    x = np.asarray(x_true); y = np.asarray(y_pred)
    diff = y - x
    mean = (y + x)/2
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_low, loa_high = md - 1.96*sd, md + 1.96*sd
    return mean, diff, md, sd, loa_low, loa_high

# ---------- plotting helpers ----------
def common_limits(*arrays):
    v = np.concatenate([np.asarray(a).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    lo, hi = np.min(v), np.max(v)
    if hi == lo:
        return (lo-1, hi+1)
    pad = 0.05*(hi - lo)
    return (lo - pad, hi + pad)

def plot_bland_altman_inset(ax, x_true, y_pred, loc="upper left", width=0.45, height=0.45, fontsize=7):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    mean, diff, md, sd, lo, hi = bland_altman_stats(x_true, y_pred)
    iax = inset_axes(ax, width=f"{int(width*100)}%", height=f"{int(height*100)}%", loc=2 if loc=="upper left" else 1)
    iax.scatter(mean, diff, s=6, alpha=0.6, edgecolor="none")
    iax.axhline(md, color="k", lw=1)
    iax.axhline(lo, color="k", lw=1, ls="--")
    iax.axhline(hi, color="k", lw=1, ls="--")
    iax.tick_params(labelsize=fontsize-1)
    iax.set_title("Bland–Altman", fontsize=fontsize)
    iax.set_xlabel("Mean", fontsize=fontsize)
    iax.set_ylabel("Diff (pred-true)", fontsize=fontsize)
    return iax

def plot_agreement(ax, x_true, y_pred, title, units="", match_limits=None,
                   lam=1.0, B=200, show_legend=False, show_ccc=False, show_ba=False):
    x = np.asarray(x_true); y = np.asarray(y_pred)

    # Deming & stats
    m, b = deming(x, y, lam=lam)
    lo, hi = deming_ci(x, y, lam=lam, B=2000)
    r2m, r2sd = bootstrap_r2_score(x, y, B=B)
    err = rmse(x, y)
    ccc_val = ccc(x, y) if show_ccc else None

    # Limits
    lims = common_limits(x, y) if match_limits is None else match_limits

    # Plots
    sc = ax.scatter(x, y, s=24, alpha=0.7, edgecolor="none", label="Predictions")
    id_line, = ax.plot(lims, lims, ls="--", lw=1.5, label="Identity (y=x)")
    xs = np.linspace(*lims, 100)
    dm_line, = ax.plot(xs, m*xs + b, lw=1.5, label="Deming fit")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f"True ({units})" if units else "True")
    ax.set_ylabel(f"Predicted ({units})" if units else "Predicted")
    ax.set_title(title, fontsize=10)

    # Metrics text
    txt = f"Deming slope={m:.2f} [{lo:.2f},{hi:.2f}]\nIntercept={b:.3f}\nRMSE={err:.3g}\n$R^2$={r2m:.2f}±{r2sd:.02f}"
    if show_ccc and ccc_val is not None and np.isfinite(ccc_val):
        txt += f"\nCCC={ccc_val:.2f}"
    ax.text(0.02, 0.98, txt, ha="left", va="top", transform=ax.transAxes,
            fontsize=8, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"))

    if show_legend:
        ax.legend(loc="lower right", fontsize=8, frameon=True)

    if show_ba:
        plot_bland_altman_inset(ax, x, y, loc="upper left", width=0.48, height=0.48, fontsize=7)

# ---------- data IO ----------
def load_from_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    return (df["kPL_true"].values, df["kPL_pred"].values,
        df["kVE_true"].values, df["kVE_pred"].values,
        df["vB_true"].values,  df["vB_pred"].values)

def main():
    ap = argparse.ArgumentParser(description="Agreement plots with Deming regression, CCC, and optional Bland–Altman insets.")
    ap.add_argument("--csv_nlls", required=True, help="CSV for NLLS (columns: kPL_true, kPL_pred, kVE_true, kVE_pred, vB_true, vB_pred)")
    ap.add_argument("--csv_nn", required=True, help="CSV for NN (same column layout as --csv_nlls)")
    ap.add_argument("--out", default="figure2_regenerated.png")
    ap.add_argument("--lam", type=float, default=1.0, help="Deming variance ratio sigma_y^2/sigma_x^2")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--show_ccc", action="store_true", help="Add Lin's CCC to metrics")
    ap.add_argument("--show_ba_insets", action="store_true", help="Add Bland–Altman inset to each panel")
    args = ap.parse_args()

    (true_kpl_nlls, pred_kpl_nlls,
     true_kve_nlls, pred_kve_nlls,
     true_vb_nlls,  pred_vb_nlls) = load_from_csv(args.csv_nlls)
    (true_kpl, pred_kpl,
     true_kve, pred_kve,
     true_vb,  pred_vb) = load_from_csv(args.csv_nn)

    # Shared limits
    lim_kpl = common_limits(true_kpl_nlls, pred_kpl_nlls, true_kpl, pred_kpl)
    lim_kve = common_limits(true_kve_nlls, pred_kve_nlls, true_kve, pred_kve)
    lim_vb  = common_limits(true_vb_nlls,  pred_vb_nlls,  true_vb,  pred_vb)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)

    # Top row: NLLS
    plot_agreement(axes[0,0], true_kpl_nlls, pred_kpl_nlls, "(A) NLLS  kPL", units="s$^{-1}$",
                   match_limits=lim_kpl, lam=args.lam, B=args.bootstrap,
                   show_legend=True, show_ccc=args.show_ccc, show_ba=args.show_ba_insets)
    plot_agreement(axes[0,1], true_kve_nlls, pred_kve_nlls, "(A) NLLS  kVE", units="s$^{-1}$",
                   match_limits=lim_kve, lam=args.lam, B=args.bootstrap,
                   show_legend=False, show_ccc=args.show_ccc, show_ba=args.show_ba_insets)
    plot_agreement(axes[0,2], true_vb_nlls,  pred_vb_nlls,  "(A) NLLS  vB",  units="",
                   match_limits=lim_vb,  lam=args.lam, B=args.bootstrap,
                   show_legend=False, show_ccc=args.show_ccc, show_ba=args.show_ba_insets)

    # Bottom row: NN
    plot_agreement(axes[1,0], true_kpl, pred_kpl, "(B) NN  kPL", units="s$^{-1}$",
                   match_limits=lim_kpl, lam=args.lam, B=args.bootstrap,
                   show_legend=True, show_ccc=args.show_ccc, show_ba=args.show_ba_insets)
    plot_agreement(axes[1,1], true_kve, pred_kve, "(B) NN  kVE", units="s$^{-1}$",
                   match_limits=lim_kve, lam=args.lam, B=args.bootstrap,
                   show_legend=False, show_ccc=args.show_ccc, show_ba=args.show_ba_insets)
    plot_agreement(axes[1,2], true_vb,  pred_vb,  "(B) NN  vB",  units="",
                   match_limits=lim_vb,  lam=args.lam, B=args.bootstrap,
                   show_legend=False, show_ccc=args.show_ccc, show_ba=args.show_ba_insets)

    for ax in axes.ravel():
        ax.tick_params(labelsize=9)

    plt.savefig(args.out, dpi=300)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
