import subprocess

import numpy as np
import os
import json
import re
import argparse
from collections import Counter
import torch
import nibabel as nib
from sklearn.metrics import r2_score
from glob import glob
from fit_two_compartment_physio_nlls import fit_nlls_gamma_AIF
from hybrid_model_utils import HybridMultiHead, compute_peak_percentile, load_training_config, prepare_hybrid_inputs
from hybrid_model_utils import (load_nifti_series, evaluate_model)

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


def _normalize_series_shape(arr, label):
    """
    Normalize loaded dynamic series to (..., T).
    Accepts both (..., T) and (..., T, 1).
    """
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError(f"{label} has invalid shape {arr.shape}; expected at least 2D with time in last axis")
    if arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim < 2:
        raise ValueError(f"{label} collapsed to invalid shape {arr.shape} after squeeze")
    return arr

def process_pair(idx, pyr_file, mask_file, lac_file, training_info_dir=None, slice_idx=None):
    
        # Load a trained model 
    weights_path = os.path.join(training_info_dir, "trained_hybrid_positive.pth")
    training_data_info_path = os.path.join(training_info_dir, "training_report.md")


    _cfg = load_training_config(training_data_info_path)
    NUM_TIMEPOINTS = _cfg["NUM_TIMEPOINTS"]
    PYR_FA_SCHEDULE = _cfg["PYR_FA_SCHEDULE"]
    LAC_FA_SCHEDULE = _cfg["LAC_FA_SCHEDULE"]
    SCAN_TR = _cfg["SCAN_TR"]  # seconds
    P_train = _cfg["P_TRAIN"]   
    percentile = _cfg["PERCENTILE"]  #
    KPL_MIN = _cfg["KPL_MIN"]
    KPL_MAX = _cfg["KPL_MAX"]
    KVE_MIN = _cfg["KVE_MIN"]
    KVE_MAX = _cfg["KVE_MAX"]
    VB_MIN = _cfg["VB_MIN"]
    VB_MAX = _cfg["VB_MAX"]
    
    from datetime import datetime
    import time



    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d-H%M%S")
    training_tag = os.path.basename(os.path.normpath(training_info_dir))
    save_root = os.path.join("output", training_tag, f"Clinical_Data_{timestamp}")
    os.makedirs(save_root, exist_ok=True)
    
    exam_id = _exam_id_from_path(pyr_file)
    print(f"\n=== Processing pair {idx} ({exam_id}): {os.path.basename(pyr_file)}, {os.path.basename(lac_file)}===")
    pair_dir = os.path.join(save_root, f"pair_{idx:02d}_{exam_id}")
    os.makedirs(pair_dir, exist_ok=True)

    # Load time-series data (T in last dim) and keep native spatial orientation.
    pyr = _normalize_series_shape(load_nifti_series([pyr_file]), "pyruvate")
    lac = _normalize_series_shape(load_nifti_series([lac_file]), "lactate")
    if pyr.shape != lac.shape:
        raise ValueError(f"Shape mismatch between pyruvate {pyr.shape} and lactate {lac.shape}")
    out_affine = nib.load(pyr_file).affine

    if pyr.shape[-1] != NUM_TIMEPOINTS:
        raise ValueError(
            f"Timepoint mismatch: data has {pyr.shape[-1]} frames but training config expects {NUM_TIMEPOINTS}"
        )

    # Optional single-slice mode: extract one z-slice for processing.
    # Results are embedded back into full-volume NaN arrays before saving.
    full_volume_shape = pyr.shape[:-1]   # spatial dims of the full volume
    if slice_idx is not None:
        if pyr.ndim < 4:
            raise ValueError(
                f"--slice requires a 3-D spatial volume (x,y,z,T), but data has shape {pyr.shape}"
            )
        n_slices = pyr.shape[2]
        if not (0 <= slice_idx < n_slices):
            raise ValueError(f"--slice {slice_idx} is out of range for a volume with {n_slices} z-slices (0–{n_slices-1})")
        print(f"Slice mode: processing z-slice {slice_idx} of {n_slices} only.")
        pyr = pyr[:, :, slice_idx, :]   # (x, y, T)
        lac = lac[:, :, slice_idx, :]

    # Compute AUCs
    # AUC := sum over time axis (last dim)
    AUC_pyr, AUC_lac, AUC_ratio = compute_auc_ratio(pyr, lac)
    # When processing a single slice, embed results into full-volume NaN arrays
    # so that output NIfTI files have the same spatial dimensions as the input.
    def _embed_slice(data_slice, full_shape, z):
        vol = np.full(full_shape, np.nan, dtype=np.float32)
        vol[:, :, z] = data_slice
        return vol

    if slice_idx is not None:
        AUC_pyr_save   = _embed_slice(AUC_pyr,   full_volume_shape, slice_idx)
        AUC_lac_save   = _embed_slice(AUC_lac,   full_volume_shape, slice_idx)
        AUC_ratio_save = _embed_slice(AUC_ratio, full_volume_shape, slice_idx)
    else:
        AUC_pyr_save, AUC_lac_save, AUC_ratio_save = AUC_pyr, AUC_lac, AUC_ratio

    # Save NIfTI volumes (use pyr as reference)
    out_pyr = os.path.join(pair_dir, "AUC_pyr.nii.gz")
    out_lac = os.path.join(pair_dir, "AUC_lac.nii.gz")
    out_ratio = os.path.join(pair_dir, "AUC_ratio.nii.gz")
    _save_param_map(AUC_pyr_save, out_affine, out_pyr)
    _save_param_map(AUC_lac_save, out_affine, out_lac)
    _save_param_map(AUC_ratio_save, out_affine, out_ratio)
    
    log = {"exam_id": exam_id}

    # === Prepare model inputs (vox x T x C -> flattened for model) ===
    pyr_2d = pyr.reshape(-1, pyr.shape[-1])
    lac_2d = lac.reshape(-1, lac.shape[-1])
    X_combined = np.stack([pyr_2d, lac_2d], axis=-1)  # (vox, T, 2)
    
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
    if not hasattr(process_pair, "_model"):
        model = HybridMultiHead(input_dim_norm=NUM_TIMEPOINTS*2, 
                        input_dim_raw=NUM_TIMEPOINTS*2,
                        vb_range=(VB_MIN, VB_MAX),
                        kpl_range=(KPL_MIN, KPL_MAX),
                        kve_range=(KVE_MIN, KVE_MAX)
                        )

        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
        model.eval()
        process_pair._model = model
    else:
        model = process_pair._model

    # === Neural Network Prediction ===
    pred = evaluate_model(model, X_norm, X_raw)
    volume_shape = pyr.shape[:-1]
    param_names = ["kPL", "kVE", "vB"]
    param_maps = {}
    for i, name in enumerate(param_names):
        param_map = pred[:, i].reshape(volume_shape)
        param_maps[name] = param_map 

    # Save pre-calibration neural network maps
    for name in param_names:
        map_to_save = (_embed_slice(param_maps[name], full_volume_shape, slice_idx)
                       if slice_idx is not None else param_maps[name])
        outpath = os.path.join(pair_dir, f"{name}_map_NN_pair{1:02d}.nii.gz")#kofi tmp for montage testing
        _save_param_map(map_to_save, out_affine, outpath)

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
    skipped_low_or_nonfinite = 0
    fit_exception_count = 0
    exception_type_counts = Counter()
    time_points=np.arange(0, NUM_TIMEPOINTS * SCAN_TR, SCAN_TR) # 16 time points from 0 to 30s with TR=2s

    nlls_bounds = (
        (KPL_MIN, KVE_MIN, VB_MIN),
        (KPL_MAX, KVE_MAX, VB_MAX)
    )

    nlls_p0 = (
        0.5 * (KPL_MIN + KPL_MAX),
        0.5 * (KVE_MIN + KVE_MAX),
        0.5 * (VB_MIN + VB_MAX),
    )

    
    for vox_idx in range(n_voxels):
        pyr_signal = pyr_2d[vox_idx, :]
        lac_signal = lac_2d[vox_idx, :]
        
        # Skip voxels with very low signal or NaN values
        if (np.max(pyr_signal) < 1e-6 or np.max(lac_signal) < 1e-6 or 
            not np.all(np.isfinite(pyr_signal)) or not np.all(np.isfinite(lac_signal))):
            skipped_low_or_nonfinite += 1
            continue
            
        try:    
            params = fit_nlls_gamma_AIF(
                time_points=time_points,
                S_pyr_obs=pyr_signal,
                S_lac_obs=lac_signal,
                TR=SCAN_TR,
                flips_pyr_deg=PYR_FA_SCHEDULE,
                flips_lac_deg=LAC_FA_SCHEDULE,
                R1p=1/30,
                R1l=1/25,
                p0=nlls_p0,
                bounds=nlls_bounds,
                sigma_pyr=None,
                sigma_lac=None,
            )
             
            # Convert flat index back to 3D coordinates
            coords = np.unravel_index(vox_idx, volume_shape)
            traditional_kpl[coords] = params["kpl"]  # kPL
            traditional_kve[coords] = params["kve"]  # kVE  
            traditional_vb[coords] = params["vb"]   # vB
            successful_fits += 1
                
        except Exception as e:
            fit_exception_count += 1
            exception_type_counts[type(e).__name__] += 1
            continue
    
    traditional_fit_time = time.time() - start_time
    print(f"Traditional fitting completed in {traditional_fit_time:.2f} seconds")
    print(f"Successful fits: {successful_fits}/{n_voxels} voxels ({100*successful_fits/n_voxels:.1f}%)")
    print(
        "Traditional fit diagnostics: "
        f"skipped_low_or_nonfinite={skipped_low_or_nonfinite}, "
        f"fit_exceptions={fit_exception_count}"
    )
    if fit_exception_count > 0:
        top_exception_types = exception_type_counts.most_common(3)
        top_exception_summary = ", ".join(
            f"{name}={count}" for name, count in top_exception_types
        )
        print(f"Top fit exception types: {top_exception_summary}")
    
    # Store traditional fitting results
    param_maps["kPL_NLLS"] = traditional_kpl
    param_maps["kVE_NLLS"] = traditional_kve
    param_maps["vB_NLLS"] = traditional_vb
    
    # Save traditional fitting maps
    for trad_name, trad_map in [("kPL_NLLS", traditional_kpl),
                                ("kVE_NLLS", traditional_kve),
                                ("vB_NLLS", traditional_vb)]:
        map_to_save = (_embed_slice(trad_map, full_volume_shape, slice_idx)
                       if slice_idx is not None else trad_map)
        outpath = os.path.join(pair_dir, f"{trad_name}_map_pair{1:02d}.nii.gz")#kofi tmp for montage testing
        _save_param_map(map_to_save, out_affine, outpath)
    
    # Update log with traditional fitting info
    log.update({
        "traditional_fitting": {
            "fitting_time_seconds": traditional_fit_time,
            "successful_fits": successful_fits,
            "total_voxels": n_voxels,
            "success_rate_percent": 100 * successful_fits / n_voxels if n_voxels > 0 else 0,
            "skipped_low_or_nonfinite": skipped_low_or_nonfinite,
            "fit_exception_count": fit_exception_count,
            "exception_type_counts": dict(exception_type_counts)
        }
    })

        
    
    
    
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
    for param_name in ["kPL_NLLS", "kVE_NLLS", "vB_NLLS"]:
        if param_name in param_maps:
            stats.update(summarize(param_name, param_maps[param_name], mask))
    if "kPL_NLLS_CAL" in param_maps:
        stats.update(summarize("kPL_NLLS_CAL", param_maps["kPL_NLLS_CAL"], mask))
        
    stats_path = os.path.join(pair_dir, "parameter_and_AUC_summary.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    log_path = os.path.join(pair_dir, "run_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print("Wrote:")
    print(" ", out_pyr)
    print(" ", out_lac)
    print(" ", out_ratio)
    print(" ", stats_path)
    print(" ", log_path)
    
    # After all maps are saved to pair_dir
    montage_output = os.path.join(pair_dir, "montage.png")
    #Add slice argument to montage command if in slice mode, to avoid montage script trying to auto-crop to non-existent slices
    
    cmd = [
            "python", "hp13c_montage.py", pair_dir, "-o", montage_output, "--auto-window"
        ]
    
    if mask_file is not None:
        cmd.extend(["--auto-crop", "--mask", mask_file])

    if slice_idx is not None:
        cmd.extend(["--slice", str(slice_idx)])
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Montage saved to: {montage_output}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: montage generation failed: {e.stderr}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run data inference using a trained hybrid model."
    )
    parser.add_argument(
        "--training_info_dir",
        required=True,
        help="Subdirectory under output/ containing trained_hybrid_positive.pth and training_report.md",
    )
    
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to  pyr*.nii.gz and lac*.nii.gz"
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        dest="slice_idx",
        metavar="Z",
        help=(
            "0-based z-slice index to process instead of the full image stack. "
            "Useful for quick testing — only that slice is run through NN and NLLS; "
            "all other slices are NaN in the output volumes."
        ),
    )

    # Ignore unrelated args from debuggers/notebooks while still enforcing required args.
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    data_dir = os.path.dirname(os.path.abspath(args.data_path))
    training_info_dir = os.path.abspath(args.training_info_dir)
    #keyword is embedded in path string,e.g. pyr*TRAMP*.nii.gz to match both pyr and lac files for the same exam
    #extract keyword from data_dir name, e.g. "TRAMP" from "data/TRAMP_experiment"
    keyword = os.path.basename(args.data_path).split("*")[1] if "*" in args.data_path else ""

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir does not exist or is not a directory: {data_dir}")
    if not os.path.isdir(training_info_dir):
        raise FileNotFoundError(
            f"training_info_dir does not exist or is not a directory: {training_info_dir}"
        )

    # === Specify dataset ===
    pyr_files_names = sorted(glob(os.path.join(data_dir, f"pyr*{keyword}*.nii.gz")))
    lac_files_names = sorted(glob(os.path.join(data_dir, f"lac*{keyword}*.nii.gz")))
    print("Found pyruvate files:")
    for f in pyr_files_names:
        print(" ", f)   
    pyr_files = sorted(glob(os.path.join(data_dir, f"pyr*{keyword}*.nii.gz")))
    lac_files = sorted(glob(os.path.join(data_dir, f"lac*{keyword}*.nii.gz")))
    mask_files = sorted(glob(os.path.join(data_dir, f"mask*{keyword}*.nii.gz")))  # optional masks
    if not pyr_files or not lac_files:
        raise FileNotFoundError(
            "Expected files not found in data_dir: pyruvate_TRAMP.nii.gz and/or lactate_TRAMP.nii.gz"
        )
    assert len(pyr_files) == len(lac_files), "Must have same number of pyr/lac files."


    for idx, (pf, lf) in enumerate(zip(pyr_files, lac_files), 1):
        process_pair(idx, pf, mask_files[idx-1] if mask_files else None, lf,
                     training_info_dir=training_info_dir, slice_idx=args.slice_idx)
