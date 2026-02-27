import numpy as np
import os
import json
import re
import torch
import nibabel as nib
from sklearn.metrics import r2_score
from glob import glob
from hybrid_model_utils import HybridMultiHead, compute_peak_percentile
from hybrid_model_utils import (load_nifti_series, evaluate_model)
from measured_pyr_driver_kpl_kve_vb_gain import fit_measured_pyr_driver_kve_vb
from hybrid_model_utils import compute_robust_alpha, prepare_hybrid_inputs


# === Specify dataset ===
# pyr_files = sorted(glob("data/pyr*Brain1.nii.gz"))
# lac_files = sorted(glob("data/lac*Brain1.nii.gz"))
# pyr_files = sorted(glob("data/pyruvate_TRAMP.nii.gz"))
# lac_files = sorted(glob("data/lactate_TRAMP.nii.gz"))
pyr_files = sorted(glob("data/pyruvateRatKidneysEPI_exp2_constant.nii.gz"))
lac_files = sorted(glob("data/lactateRatKidneysEPI_exp2_constant.nii.gz"))
assert len(pyr_files) == len(lac_files), "Must have same number of pyr/lac files."

# Load a trained model 
# weights_dir = "wts_2C_MeasuredAIF_noisestd0.05"
weights_dir = "noiselevel_0.05_20260226-153922"
weights_path = os.path.join("output", weights_dir, "trained_hybrid_positive.pth")
training_data_info_path = os.path.join("output", weights_dir, "training_report.md")


def _parse_schedule(text, key):
    match = re.search(rf"{key}:\s*\[([^\]]+)\]", text)
    if not match:
        return None
    vals = [v.strip() for v in match.group(1).split(',') if v.strip()]
    if not vals:
        return None
    try:
        return [float(v) for v in vals]
    except ValueError:
        return None


def _load_training_config(path):
    """Load basic timing/flip info from the training report if present."""
    defaults = {
        "NUM_TIMEPOINTS": 12,
        "PYR_FA_SCHEDULE": [11.0] * 12,
        "LAC_FA_SCHEDULE": [80.0] * 12,
        "SCAN_TR": 5.0,
        "TRAINING_PEAK": 0.151621,
    }
    if not os.path.exists(path):
        print(f"Warning: training info not found at {path}; using defaults.")
        return defaults

    with open(path, "r") as f:
        text = f.read()

    cfg = dict(defaults)

    m = re.search(r"NUM_TIME_POINTS:\s*([0-9]+)", text)
    if m:
        cfg["NUM_TIMEPOINTS"] = int(m.group(1))

    m = re.search(r"SCAN_TR\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        try:
            cfg["SCAN_TR"] = float(m.group(1))
        except ValueError:
            pass

    sched = _parse_schedule(text, "PYR_FA_SCHEDULE")
    if sched:
        cfg["PYR_FA_SCHEDULE"] = sched

    sched = _parse_schedule(text, "LAC_FA_SCHEDULE")
    if sched:
        cfg["LAC_FA_SCHEDULE"] = sched

    m = re.search(r"TRAINING_PEAK:\s*([0-9eE\.+-]+)", text)
    if m:
        try:
            cfg["TRAINING_PEAK"] = float(m.group(1))
        except ValueError:
            pass

    m = re.search(r"P_train:\s*([0-9eE\.+-]+)", text)
    if m:
        try:
            cfg["P_TRAIN"] = float(m.group(1))
        except ValueError:
            pass
        
    m = re.search(r"percentile:\s*([0-9eE\.+-]+)", text)
    if m:
        try:
            cfg["PERCENTILE"] = float(m.group(1))
        except ValueError:
            pass
        
    return cfg


_cfg = _load_training_config(training_data_info_path)
NUM_TIMEPOINTS = _cfg["NUM_TIMEPOINTS"]
PYR_FA_SCHEDULE = _cfg["PYR_FA_SCHEDULE"]
LAC_FA_SCHEDULE = _cfg["LAC_FA_SCHEDULE"]
SCAN_TR = _cfg["SCAN_TR"]  # seconds
# TRAINING_PEAK = _cfg["TRAINING_PEAK"]
P_train = _cfg["P_TRAIN"]   
percentile = _cfg["PERCENTILE"]  #
TRAINING_PEAK = 1
FINE_TUNE_FACTOR = 1  # If >0, we can apply a modest additional scaling to the inputs to better match the training peak, without losing the sim→clinic consistency of the original robust peak scaling. The model can learn to adjust for this during fine-tuning, and it can help if your data's robust peak is systematically much higher or lower than the training peak.

from scipy.ndimage import gaussian_filter
from datetime import datetime
import time

# ===== Additional NAWM calibrations for kVE and vB =====
enable_nawm_calibration_kve = True
enable_nawm_calibration_vb  = True

# Option B: load cohort-derived targets, if you’ve computed them before
# File format (JSON): {"kPL": 0.0175, "kVE": 0.23, "vB": 0.018}
cohort_targets_json = None  # e.g., "calibration_targets_nawm.json"



# Optional NAWM masks (same count as exams; if absent, NAWM calibration is skipped)
nawm_masks = sorted(glob("data/*nawm*1.nii.gz"))
use_nawm = len(nawm_masks) == len(pyr_files)

# Optional VIF amplitudes CSV: columns: exam_id, vif_amp
vif_csv_path = "data/vif_amplitudes.csv"


# Output directory
timestamp = datetime.now().strftime("%Y%m%d-H%M%S")
save_root = os.path.join("output", weights_dir, f"Clinical_Data_{timestamp}")
os.makedirs(save_root, exist_ok=True)

# === Amplitude normalization settings (sim→clinic consistent) ===
# Choose one: "robust_peak" or "vif_amp"
amplitude_norm_mode = "hybrid_norm"  # "robust_peak", "vif_amp", or None to disable amplitude normalization

# If using robust peak: which channels contribute to the peak?
include_bic_in_peak = False  # typical: pyr + lac only
percentile = 99.9            # robust high-percentile peak



# If training used an extra scale like "peak*2", set legacy_peak_divisor=2.0; otherwise 1.0
legacy_peak_divisor = 1.0

# If using VIF amplitude: the model expects signals scaled so that VIF amplitude ~= training_vif_amp
# (Set training_vif_amp to the amplitude used during training; here we mirror training_peak for simplicity.)
training_vif_amp = TRAINING_PEAK

# === NAWM single-parameter calibration (scale-only) ===
enable_nawm_calibration = False
nawm_target_mean = 0.0175   # s^-1, midpoint of 0.015–0.02
# If no NAWM mask supplied for an exam, calibration for that exam is skipped gracefully.

# Option A: fixed targets (edit to your preference or replace with cohort targets)
# NOTE: These are reasonable starting points for brain NAWM; adjust to your protocol/model.
nawm_target_mean_kve = 0.25   # s^-1 (example)
nawm_target_mean_vb  = 0.02   # fraction (2%)
# ===== Reference-tissue (auto-ROI) calibration, for non-brain =====
enable_auto_reference_calibration = True
# auto_ref_target_mean = 0.0175  # s^-1; tweak to 0.015–0.02 for mouse skeletal muscle
auto_ref_target_mean = 0.02 # s^-1; tweak to 0.015–0.02 for mouse skeletal muscle
auto_ref_lowR_percentile = 10.0  # use lowest 10% Lac/Pyr voxels as "muscle-like"
auto_ref_vessel_exclude_pct = 99.0  # drop top 1% AUC_pyr (vessels)
auto_ref_min_AUCp = 1e-4  # tiny floor to avoid division noise

def _auc_ratio_masks(pyr, lac, vessel_exclude_pct=99.0, lowR_pct=10.0, min_aucp=1e-4):
    # pyr, lac: (X,Y,Z,T)
    AUCp = np.sum(pyr, axis=-1)
    AUCl = np.sum(lac, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        R = AUCl / np.maximum(AUCp, 1e-12)

    # exclude obvious vessels by AUCp
    vessel_thr = np.percentile(AUCp[np.isfinite(AUCp)], vessel_exclude_pct)
    not_vessel = AUCp <= vessel_thr

    # basic validity mask
    valid = np.isfinite(R) & np.isfinite(AUCp) & (AUCp > min_aucp)

    # candidate mask
    cand = valid & not_vessel
    if np.count_nonzero(cand) < 50:  # too few voxels; bail
        return cand, np.zeros_like(R, dtype=bool), R, AUCp

    # pick lowest lowR_pct% of R within candidates
    R_cand = R[cand]
    cut = np.percentile(R_cand, lowR_pct)
    lowR_mask = cand & (R <= cut)
    return cand, lowR_mask, R, AUCp

def _auto_reference_calibrate_kpl(pyr, lac, kpl_map, target_mean, logdict, cfg):
    cand, lowR_mask, R, AUCp = _auc_ratio_masks(
        pyr, lac,
        vessel_exclude_pct=cfg.get("auto_ref_vessel_exclude_pct", 99.0),
        lowR_pct=cfg.get("auto_ref_lowR_percentile", 10.0),
        min_aucp=cfg.get("auto_ref_min_AUCp", 1e-4),
    )
    if np.count_nonzero(lowR_mask) < 50:
        logdict["auto_ref"] = {"warning": "Too few auto-ROI voxels; skipped."}
        return kpl_map, logdict

    kpl_vals = kpl_map[lowR_mask]
    kpl_vals = kpl_vals[np.isfinite(kpl_vals)]
    if kpl_vals.size < 50:
        logdict["auto_ref"] = {"warning": "Too few finite kPL in auto-ROI; skipped."}
        return kpl_map, logdict

    med_pre = float(np.median(kpl_vals))
    s = target_mean / max(med_pre, 1e-12)
    kpl_cal = kpl_map * s
    med_post = float(np.median(kpl_cal[lowR_mask]))
    logdict["auto_ref"] = {
        "auto_ref_voxels": int(kpl_vals.size),
        "auto_ref_median_pre": med_pre,
        "auto_ref_target_mean": float(target_mean),
        "calibration_scale": float(s),
        "auto_ref_median_post": med_post
    }
    return kpl_cal, logdict

def _median_target_scale(vol, mask, target):
    vals = vol[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size < 50:  # need enough voxels for a stable median
        return None, None, "Too few voxels for stable calibration."
    med = float(np.median(vals))
    s = target / max(med, 1e-12)
    return s, med, None

def _load_cohort_targets(json_path, fallback):
    if not json_path or not os.path.exists(json_path):
        return fallback
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        out = dict(fallback)
        for k in ["kPL","kVE","vB"]:
            if k in d and isinstance(d[k], (int, float)):
                out[k] = float(d[k])
        return out
    except Exception:
        return fallback
    
    
def _load_vif_amplitudes(csv_path):
    if not os.path.exists(csv_path):
        return {}
    table = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        # expect exam_id, vif_amp
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 2:
                continue
            exam_id, amp_str = parts[0], parts[1]
            try:
                table[exam_id] = float(amp_str)
            except ValueError:
                continue
    return table

def _exam_id_from_path(p):
    base = os.path.basename(p)
    # drop common extensions like .nii.gz
    if base.endswith(".nii.gz"):
        base = base[:-7]
    else:
        base = os.path.splitext(base)[0]
    # also strip leading "pyr" if present to get a cleaner identifier
    if base.startswith("pyr"):
        base = base[3:]
    return base

def _compute_robust_peak(pyr, lac, percentile=99.9, include_bic=False):
    if include_bic:
        combined = np.concatenate([pyr.flatten(), lac.flatten()])
    else:
        combined = np.concatenate([pyr.flatten(), lac.flatten()])
    return np.percentile(combined, percentile)

def _apply_amplitude_normalization(pyr, lac, mode, robust_peak=None, vif_amp=None):
    """
    Returns normalized (pyr, lac, bic), along with the scaling factor applied.
    """
    if mode == "robust_peak":
        assert robust_peak is not None and robust_peak > 0, "robust_peak must be provided for robust_peak mode."
        # Match training: scale so that robust_peak / legacy_divisor == training_peak
        # => scale = training_peak / (robust_peak / legacy_divisor) = training_peak*legacy_divisor/robust_peak
        scale = (TRAINING_PEAK * legacy_peak_divisor) / max(robust_peak, 1e-12)
    elif mode == "vif_amp":
        assert vif_amp is not None and vif_amp > 0, "vif_amp must be provided for vif_amp mode."
        # Scale to make VIF amplitude match the training VIF amplitude
        scale = training_vif_amp / max(vif_amp, 1e-12)
    else:
        raise ValueError(f"Unknown amplitude_norm_mode: {mode}")

    pyr_peaks = np.max(pyr, axis=-1, keepdims=True)
    # 2. Prevent division by zero for background voxels
    pyr_peaks[pyr_peaks == 0] = 1.0 
    # 3. Normalize everything at once!
    pyr_norm = pyr / pyr_peaks
    lac_norm = lac / pyr_peaks  
    scale = 1    
    
    return pyr_norm*scale, lac_norm*scale, float(scale)

def summarize(name, vol, mask=None, pcts=(1,5,25,50,75,95,99)):
    v = vol[np.isfinite(vol)]
    if mask is not None:
        v = vol[(mask.astype(bool)) & np.isfinite(vol)]
    if v.size == 0:
        return {f"{name}_n": 0}
    stats = {
        f"{name}_n": int(v.size),
        f"{name}_min": float(np.min(v)),
        f"{name}_max": float(np.max(v)),
        f"{name}_mean": float(np.mean(v)),
        f"{name}_median": float(np.median(v)),
        f"{name}_std": float(np.std(v)),
    }
    for p in pcts:
        stats[f"{name}_p{p}"] = float(np.percentile(v, p))
    return stats

def _save_param_map(param_volume, affine, outpath):
    img = nib.Nifti1Image(param_volume.astype(np.float32), affine=affine)
    nib.save(img, outpath)
    print("Saved:", outpath)
    
    
def compute_auc_ratio(pyr, lac):
    """
    pyr, lac: arrays shaped (N, T) or (N, T, ) for per-sample timecourses.
    Returns:
      AUCp (N,), AUCl (N,), R (N,) where R = AUCl/AUCp
    """
    AUCp = np.sum(pyr, axis=-1)
    AUCl = np.sum(lac, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        R = AUCl / np.maximum(AUCp, 1e-12)
    return AUCp, AUCl, R

def process_pair(idx, pyr_file, lac_file, nawm_file=None, vif_amp_lookup=None):
    exam_id = _exam_id_from_path(pyr_file)
    print(f"\n=== Processing pair {idx} ({exam_id}): {os.path.basename(pyr_file)}, {os.path.basename(lac_file)}===")
    pair_dir = os.path.join(save_root, f"pair_{idx:02d}_{exam_id}")
    os.makedirs(pair_dir, exist_ok=True)

    # Load time-series data (T in last dim)
    pyr = load_nifti_series([pyr_file]).squeeze(-1)
    lac = load_nifti_series([lac_file]).squeeze(-1)
    # bic = load_nifti_series([bic_file]).squeeze()

    # Use identity affine for outputs (adjust if you prefer to inherit from one of the inputs)
    out_affine = np.eye(4)

    # Compute AUCs
    # AUC := sum over time axis (last dim)
    AUC_pyr, AUC_lac, AUC_ratio = compute_auc_ratio(pyr, lac)
    # Save NIfTI volumes (use pyr as reference)
    out_pyr = os.path.join(pair_dir, "AUC_pyr.nii.gz")
    out_lac = os.path.join(pair_dir, "AUC_lac.nii.gz")
    out_ratio = os.path.join(pair_dir, "AUC_ratio.nii.gz")
    _save_param_map(AUC_pyr, out_affine, out_pyr)
    _save_param_map(AUC_lac, out_affine, out_lac)
    _save_param_map(AUC_ratio, out_affine, out_ratio)
    
    # === Amplitude normalization (sim→clinic consistency) ===
    log = {"exam_id": exam_id, "amplitude_norm_mode": amplitude_norm_mode}

    if amplitude_norm_mode == "robust_peak":
        rp = _compute_robust_peak(pyr, lac, percentile=percentile, include_bic=include_bic_in_peak)
        pyr, lac, scale = _apply_amplitude_normalization(pyr, lac, "robust_peak", robust_peak=rp)
        log.update({"robust_peak_percentile": percentile,
                    "include_bic_in_peak": include_bic_in_peak,
                    "robust_peak_value": float(rp),
                    "legacy_peak_divisor": float(legacy_peak_divisor),
                    "applied_scale": float(scale),
                    "training_peak": float(TRAINING_PEAK)})
    elif amplitude_norm_mode == "vif_amp":  # VIF-based
        vif_amp = None
        if vif_amp_lookup is not None:
            vif_amp = vif_amp_lookup.get(exam_id)
        pyr, lac, scale = _apply_amplitude_normalization(pyr, lac, "vif_amp", vif_amp=vif_amp)
        log.update({"vif_amp_input": float(vif_amp) if vif_amp is not None else None,
                    "training_vif_amp": float(training_vif_amp),
                    "applied_scale": float(scale)})

    # === Prepare model inputs (vox x T x C -> flattened for model) ===
    pyr_2d = pyr.reshape(-1, pyr.shape[-1])
    lac_2d = lac.reshape(-1, lac.shape[-1])
    # bic_2d = bic.reshape(-1, bic.shape[-1])
    # X_combined = np.stack([pyr_2d, lac_2d, bic_2d], axis=-1)  # (vox, T, 3)
    X_combined = np.stack([pyr_2d, lac_2d], axis=-1)  # (vox, T, 3)
    
    P_clin = compute_peak_percentile(
        X_combined,        # shape (..., T, 2)
        percentile=percentile,
        pyr_channel=0,
        min_peak=1e-6
    )
    
    alpha = P_train / max(P_clin, 1e-8)

    # X_raw = X_combined.reshape(X_combined.shape[0], -1).astype(np.float32)
    # X_norm = X_raw.copy().astype(np.float32)
    
    X_norm, X_raw, clin_meta = prepare_hybrid_inputs(
        X_combined,
        alpha=alpha,
        pyr_channel=0,
        flatten=True
    )

    # === Lazy model init ===
    global model
    if not hasattr(process_pair, "_model"):
        # model = HybridMultiHead(input_dim_raw=X_raw.shape[1], input_dim_norm=X_norm.shape[1], predict_kpb=True)
        model = HybridMultiHead(input_dim_raw=NUM_TIMEPOINTS*2, input_dim_norm=NUM_TIMEPOINTS*2)
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
        model.eval()
        process_pair._model = model
    else:
        model = process_pair._model

    # === Neural Network Prediction ===
    pred = evaluate_model(model, X_norm, X_raw)
    volume_shape = pyr.shape[:-1]
    # param_names = ["kPL", "kVE", "vB", "kPB"]
    param_names = ["kPL", "kVE", "vB"]
    param_maps = {}
    for i, name in enumerate(param_names):
        param_map = pred[:, i].reshape(volume_shape)
        param_maps[name] = param_map * FINE_TUNE_FACTOR

    # Save pre-calibration neural network maps
    for name in param_names:
        outpath = os.path.join(pair_dir, f"{name}_map_NN_PRECAL_pair{idx:02d}.nii.gz")
        _save_param_map(param_maps[name], out_affine, outpath)

    # === Traditional Fitting ===
    print("Running traditional two-compartment model fitting...")
    start_time = time.time()
    
    # Prepare traditional fitting results arrays
    traditional_kpl = np.full(volume_shape, np.nan)
    traditional_kve = np.full(volume_shape, np.nan)
    traditional_vb = np.full(volume_shape, np.nan)
    
    # Get 2D versions for voxel-wise fitting
    pyr_2d = pyr.reshape(-1, pyr.shape[-1])
    lac_2d = lac.reshape(-1, lac.shape[-1])
    
    # Fit each voxel
    n_voxels = pyr_2d.shape[0]
    successful_fits = 0
    time_points=np.arange(0, NUM_TIMEPOINTS * SCAN_TR, SCAN_TR) # 16 time points from 0 to 30s with TR=2s


    
    for vox_idx in range(n_voxels):
        pyr_signal = pyr_2d[vox_idx, :]
        lac_signal = lac_2d[vox_idx, :]
        
        # Skip voxels with very low signal or NaN values
        if (np.max(pyr_signal) < 1e-6 or np.max(lac_signal) < 1e-6 or 
            not np.all(np.isfinite(pyr_signal)) or not np.all(np.isfinite(lac_signal))):
            continue
            
        try:
            params = fit_measured_pyr_driver_kve_vb(
                time_points=time_points, S_pyr=pyr_signal, S_lac=lac_signal,
                TR=SCAN_TR,
                flips_pyr_deg=PYR_FA_SCHEDULE, flips_lac_deg=LAC_FA_SCHEDULE,
                R1p=1/30.0, R1l=1/25.0,
                estimate_R1l=False,
                smooth_p_driver=True,        # light causal smoothing of S_pyr (optional)
                vb_bounds=(0.01, 0.5),       # sensible bounds help identifiability
                kve_bounds=(0.001, 1.0)
            )
             
            # Convert flat index back to 3D coordinates
            coords = np.unravel_index(vox_idx, volume_shape)
            traditional_kpl[coords] = params["kpl"]  # kPL
            traditional_kve[coords] = params["kve"]  # kVE  
            traditional_vb[coords] = params["vb"]   # vB
            successful_fits += 1
                
        except Exception as e:
            # Skip failed fits silently
            continue
    
    traditional_fit_time = time.time() - start_time
    print(f"Traditional fitting completed in {traditional_fit_time:.2f} seconds")
    print(f"Successful fits: {successful_fits}/{n_voxels} voxels ({100*successful_fits/n_voxels:.1f}%)")
    
    # Store traditional fitting results
    param_maps["kPL_TRAD"] = traditional_kpl
    param_maps["kVE_TRAD"] = traditional_kve
    param_maps["vB_TRAD"] = traditional_vb
    
    # Save traditional fitting maps
    for trad_name, trad_map in [("kPL_TRAD", traditional_kpl), 
                                ("kVE_TRAD", traditional_kve), 
                                ("vB_TRAD", traditional_vb)]:
        outpath = os.path.join(pair_dir, f"{trad_name}_map_PRECAL_pair{idx:02d}.nii.gz")
        _save_param_map(trad_map, out_affine, outpath)
    
    # Update log with traditional fitting info
    log.update({
        "traditional_fitting": {
            "fitting_time_seconds": traditional_fit_time,
            "successful_fits": successful_fits,
            "total_voxels": n_voxels,
            "success_rate_percent": 100 * successful_fits / n_voxels if n_voxels > 0 else 0
        }
    })

    # --- Auto reference-tissue calibration (if no NAWM) ---
    cfg = dict(auto_ref_vessel_exclude_pct=auto_ref_vessel_exclude_pct,
            auto_ref_lowR_percentile=auto_ref_lowR_percentile,
            auto_ref_min_AUCp=auto_ref_min_AUCp)
    if enable_auto_reference_calibration and ("kPL_CAL" not in param_maps):  # don't double-calibrate if NAWM already applied
        kpl_auto, log = _auto_reference_calibrate_kpl(
            pyr, lac, param_maps["kPL"], auto_ref_target_mean, log, cfg
        )
        if not np.array_equal(kpl_auto, param_maps["kPL"]):
            param_maps["kPL_CAL"] = kpl_auto
            _save_param_map(kpl_auto, out_affine, os.path.join(pair_dir, f"kPL_map_predicted_POSTCAL_pair{idx:02d}.nii.gz"))

    # === NAWM calibration (scale-only on kPL) ===
    nawm_stats = {}
    if enable_nawm_calibration and nawm_file is not None and os.path.exists(nawm_file):
        nawm_mask = load_nifti_series([nawm_file]).squeeze().astype(bool)
        kpl = param_maps["kPL"]
        if nawm_mask.shape != kpl.shape:
            # try to broadcast last dim or fallback
            raise ValueError(f"NAWM mask shape {nawm_mask.shape} does not match param map shape {kpl.shape}")
        nawm_vals = kpl[nawm_mask]
        nawm_vals = nawm_vals[np.isfinite(nawm_vals)]
        if nawm_vals.size > 10:
            med = float(np.median(nawm_vals))
            # scale-only: kPL_cal = s * kPL, so that median_NAWM matches target
            s = nawm_target_mean / max(med, 1e-12)
            kpl_cal = kpl * s
            param_maps["kPL_CAL"] = kpl_cal
            # Save calibrated kPL
            outpath = os.path.join(pair_dir, f"kPL_map_predicted_POSTCAL_pair{idx:02d}.nii.gz")
            _save_param_map(kpl_cal, out_affine, outpath)
            nawm_stats = {
                "nawm_voxels": int(nawm_vals.size),
                "nawm_median_pre": med,
                "nawm_target_mean": float(nawm_target_mean),
                "calibration_scale": float(s),
                "nawm_median_post": float(np.median(kpl_cal[nawm_mask]))
            }
        else:
            nawm_stats = {"warning": "NAWM mask had too few voxels; skipping calibration."}
    else:
        nawm_stats = {"info": "NAWM calibration disabled or mask not provided; skipped."}

            # ---- Load target means (prefer cohort JSON if provided) ----
    _defaults = {
        "kPL": nawm_target_mean,           # you already defined this earlier
        "kVE": nawm_target_mean_kve,
        "vB" : nawm_target_mean_vb,
    }
    targets = _load_cohort_targets(cohort_targets_json, _defaults)
    
     # ---- NAWM calibration for kVE and vB (scale-only, like kPL) ----
    if enable_nawm_calibration and nawm_file is not None and os.path.exists(nawm_file):
        nawm_mask = load_nifti_series([nawm_file]).squeeze().astype(bool)
        # shape check vs maps
        for pname in ["kVE", "vB"]:
            if pname in param_maps and param_maps[pname].shape == nawm_mask.shape:
                do_flag = (pname == "kVE" and enable_nawm_calibration_kve) or \
                          (pname == "vB"  and enable_nawm_calibration_vb)
                if do_flag:
                    s, med_pre, err = _median_target_scale(param_maps[pname], nawm_mask, targets[pname])
                    if s is not None:
                        pm_cal = param_maps[pname] * s
                        param_maps[f"{pname}_CAL"] = pm_cal
                        outpath = os.path.join(pair_dir, f"{pname}_map_predicted_POSTCAL_pair{idx:02d}.nii.gz")
                        _save_param_map(pm_cal, out_affine, outpath)
                        log.setdefault("nawm_calibration_extra", {})[pname] = {
                            "nawm_voxels": int(np.count_nonzero(nawm_mask)),
                            "median_pre": float(med_pre),
                            "target_mean": float(targets[pname]),
                            "calibration_scale": float(s),
                            "median_post": float(np.median(pm_cal[nawm_mask]))
                        }
                    else:
                        log.setdefault("nawm_calibration_extra", {})[pname] = {"warning": err}
            else:
                log.setdefault("nawm_calibration_extra", {})[pname] = {"warning": "Missing map or shape mismatch."}
        # Save a small per-exam log
        with open(os.path.join(pair_dir, "inference_log_extra.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
    else:
        log["nawm_calibration_extra"] = {"info": "Skipped (no NAWM or disabled)."}



    # === Apply same calibration to traditional fitting results ===
    if "kPL_CAL" in param_maps and "kPL_TRAD" in param_maps:
        # Use the same calibration scale that was applied to neural network results
        if enable_nawm_calibration and nawm_file is not None and os.path.exists(nawm_file):
            # Apply NAWM calibration scale to traditional results
            kpl_trad_cal = param_maps["kPL_TRAD"] * s
            param_maps["kPL_TRAD_CAL"] = kpl_trad_cal
            outpath = os.path.join(pair_dir, f"kPL_TRAD_map_POSTCAL_pair{idx:02d}.nii.gz")
            _save_param_map(kpl_trad_cal, out_affine, outpath)
            
        elif enable_auto_reference_calibration and "kPL_CAL" in param_maps:
            # Apply auto-reference calibration scale to traditional results
            # Extract scale from neural network calibration
            scale_nn = np.nanmean(param_maps["kPL_CAL"]) / np.nanmean(param_maps["kPL"])
            if np.isfinite(scale_nn) and scale_nn > 0:
                kpl_trad_cal = param_maps["kPL_TRAD"] * scale_nn
                param_maps["kPL_TRAD_CAL"] = kpl_trad_cal
                outpath = os.path.join(pair_dir, f"kPL_TRAD_map_POSTCAL_pair{idx:02d}.nii.gz")
                _save_param_map(kpl_trad_cal, out_affine, outpath)

    # Save a small per-exam log
    log.update({"nawm_calibration": nawm_stats})
    with open(os.path.join(pair_dir, "inference_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
        
    
    
    
     # Stats JSON
    mask = None
    stats = {}
    stats.update(summarize("AUC_pyr", AUC_pyr, mask))
    stats.update(summarize("AUC_lac", AUC_lac, mask))
    stats.update(summarize("AUC_ratio", AUC_ratio, mask))
    
    # Add statistics for neural network parameters
    for param_name in ["kPL", "kVE", "vB"]:
        if param_name in param_maps:
            stats.update(summarize(f"NN_{param_name}", param_maps[param_name], mask))
        if f"{param_name}_CAL" in param_maps:
            stats.update(summarize(f"NN_{param_name}_CAL", param_maps[f"{param_name}_CAL"], mask))
    
    # Add statistics for traditional fitting parameters
    for param_name in ["kPL_TRAD", "kVE_TRAD", "vB_TRAD"]:
        if param_name in param_maps:
            stats.update(summarize(param_name, param_maps[param_name], mask))
    if "kPL_TRAD_CAL" in param_maps:
        stats.update(summarize("kPL_TRAD_CAL", param_maps["kPL_TRAD_CAL"], mask))
        
    stats_path = os.path.join(pair_dir, "parameter_and_AUC_summary.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Wrote:")
    print(" ", out_pyr)
    print(" ", out_lac)
    print(" ", out_ratio)
    print(" ", stats_path)

if __name__ == "__main__":
    # Optional VIF amplitudes map
    vif_lookup = _load_vif_amplitudes(vif_csv_path) if amplitude_norm_mode == "vif_amp" else None

    if use_nawm:
        for idx, (pf, lf, nf) in enumerate(zip(pyr_files, lac_files, nawm_masks), 1):
            process_pair(idx, pf, lf, nawm_file=nf, vif_amp_lookup=vif_lookup)
    else:
        for idx, (pf, lf) in enumerate(zip(pyr_files, lac_files), 1):
            process_pair(idx, pf, lf, nawm_file=None, vif_amp_lookup=vif_lookup)
