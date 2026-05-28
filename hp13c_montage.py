"""
hp13c_montage.py
================
Build a publication-quality 3x4 montage of hyperpolarized 13C MRI maps
(AUC pyruvate / lactate / ratio plus NLLS-vs-NN kinetic parameter
estimates) from a folder of NIfTI files.

Layout
------
    [ anat / blank ]   AUC Pyruvate   AUC Lactate    AUC Ratio
    [   (B) NLLS   ]   vB (NLLS)      kVE (NLLS)     kPL (NLLS)
    [   (C) NN     ]   vB (NN)        kVE (NN)       kPL (NN)

Usage
-----
    python hp13c_montage.py /path/to/maps -o figure.png
    python hp13c_montage.py maps/ -o fig.pdf --slice 4 -w kPL=0,0.04

NLLS and NN maps of the same parameter share window/level values so
visual comparison is meaningful -- see the WINDOWS dict below to edit
defaults, or pass -w PARAM=LO,HI on the command line.

Dependencies
------------
    pip install nibabel matplotlib numpy
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================== #
#  EDIT HERE -- window/level (vmin, vmax) per parameter.
#  NLLS and NN maps of the same parameter share these values.
# =============================================================== #
WINDOWS = {
    "AUC_pyr":   (1.0,  12.0),
    "AUC_lac":   (1.0,   7.0),
    "AUC_ratio": (0.3,   2.5),
    "vB":        (0.0,   0.25),
    "kVE":       (0.05,  0.5),
    "kPL":       (0.0,   0.035),
}

CMAP = "jet"   # colormap for parametric maps


# =============================================================== #
#  Filename -> (param_id, display_label, units)
#  param_id is the key into WINDOWS so paired maps share scaling.
# =============================================================== #
FILE_MAP = {
    "AUC_pyr.nii.gz":             ("AUC_pyr",   "AUC Pyruvate", ""),
    "AUC_lac.nii.gz":             ("AUC_lac",   "AUC Lactate",  ""),
    "AUC_ratio.nii.gz":           ("AUC_ratio", "AUC Ratio",    ""),
    "vB_NLLS_map_pair01.nii.gz":  ("vB",        "vB (NLLS)",    ""),
    "kVE_NLLS_map_pair01.nii.gz": ("kVE",       "kVE (NLLS)",   r"s$^{-1}$"),
    "kPL_NLLS_map_pair01.nii.gz": ("kPL",       "kPL (NLLS)",   r"s$^{-1}$"),
    "vB_map_NN_pair01.nii.gz":    ("vB",        "vB (NN)",      ""),
    "kVE_map_NN_pair01.nii.gz":   ("kVE",       "kVE (NN)",     r"s$^{-1}$"),
    "kPL_map_NN_pair01.nii.gz":   ("kPL",       "kPL (NN)",     r"s$^{-1}$"),
}


# =============================================================== #
#  I/O
# =============================================================== #
def load_slice(path, slice_idx=None, axis=2, rot=1):
    """Return a 2D slice from a 3D or 4D NIfTI volume.

    Parameters
    ----------
    path : str | Path
        NIfTI file (.nii or .nii.gz).
    slice_idx : int or None
        Index along ``axis``. ``None`` selects the middle slice.
    axis : int
        Slicing axis (typically 2 = axial).
    rot : int
        Number of 90 deg CCW rotations applied to the 2D slice for
        display orientation. Set 0 to disable.
    """
    img = nib.load(str(path))
    data = np.squeeze(np.asanyarray(img.dataobj).astype(np.float32))

    # 4D time-series: take the first volume
    if data.ndim == 4:
        data = data[..., 0]

    # Single-slice maps come in as (X, Y, 1) and become 2D after
    # squeeze -- that 2D array IS the slice we want.
    if data.ndim == 2:
        sl = data
    elif data.ndim == 3:
        if slice_idx is None:
            slice_idx = data.shape[axis] // 2
        sl = np.take(data, slice_idx, axis=axis)
    else:
        raise ValueError(
            f"{Path(path).name}: cannot interpret shape {data.shape}"
        )

    return np.rot90(sl, k=rot) if rot else sl


def apply_mask(slc, mask):
    """Zero-out voxels of ``slc`` outside the binary mask.

    Treats any non-zero, non-NaN mask voxel as "inside". Raises if the
    mask shape doesn't match the slice -- regridding is intentionally
    not done here so silent geometry mismatches don't slip through.
    """
    if mask is None:
        return slc
    if mask.shape != slc.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match map shape {slc.shape}; "
            "resample the mask onto the metabolic grid before passing it in"
        )
    binmask = np.nan_to_num(mask, nan=0.0) > 0
    return slc * binmask


def compute_mask_bbox(mask, pad=2):
    """Tightest (x0, x1, y0, y1) box around nonzero mask voxels.

    Coordinates are in display-pixel space (x = column, y = row, origin
    top-left), with x1/y1 *exclusive* so the tuple slots straight into
    numpy slicing as ``slc[y0:y1, x0:x1]``. ``pad`` widens the box on
    every side, clipped to image bounds. Returns None for an empty mask.
    """
    if mask is None:
        return None
    binmask = np.nan_to_num(mask, nan=0.0) > 0
    if not binmask.any():
        return None

    rows_any = np.any(binmask, axis=1)
    cols_any = np.any(binmask, axis=0)
    y0, y1 = int(np.argmax(rows_any)), int(len(rows_any) - np.argmax(rows_any[::-1]))
    x0, x1 = int(np.argmax(cols_any)), int(len(cols_any) - np.argmax(cols_any[::-1]))

    H, W = binmask.shape
    return (
        max(0, x0 - pad), min(W, x1 + pad),
        max(0, y0 - pad), min(H, y1 + pad),
    )


def crop_slice(slc, bbox):
    """Apply a display-space (x0, x1, y0, y1) crop. No-op if bbox is None."""
    if bbox is None:
        return slc
    x0, x1, y0, y1 = bbox
    return slc[y0:y1, x0:x1]


def compute_auto_window(values, lo_pct=0.5, hi_pct=99.5):
    """Percentile-based window/level, in the spirit of ImageJ's Auto W/L.

    Returns ``(vmin, vmax)`` from finite values in ``values``. Falls
    back to ``(min, max)`` if the percentile range collapses, and
    finally widens by 1 if even that is degenerate so matplotlib
    doesn't choke on vmin == vmax.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            hi = lo + 1.0
    return (lo, hi)


# =============================================================== #
#  Plot helpers
# =============================================================== #
def _strip_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _draw_panel(ax, fig, slc, vmin, vmax, label, units):
    """Draw a parametric-map panel with a colorbar to its right."""
    im = ax.imshow(
        slc, cmap=CMAP, vmin=vmin, vmax=vmax,
        aspect="equal", interpolation="nearest",
    )
    _strip_axes(ax)

    # Colorbar that doesn't squash the image axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=8)
    if units:
        cax.set_title(units, fontsize=8, pad=2)

    ax.set_title(label, fontsize=11, y=-0.13)


def _draw_label_cell(ax, text):
    """Draw a black cell with white centered text, e.g. '(B) NLLS'."""
    _strip_axes(ax)
    ax.set_facecolor("black")
    ax.text(
        0.5, 0.5, text, color="white",
        ha="center", va="center", fontsize=22,
        transform=ax.transAxes,
    )


def _draw_anat(ax, anat_path, slice_idx, axis, rot, bbox=None):
    """Top-left cell: anatomical greyscale, or black if not provided.

    ``bbox`` is applied only if the anat slice has dimensions large
    enough to contain it -- avoids garbling a high-res T1 with a
    bbox computed on the low-res metabolic grid.
    """
    _strip_axes(ax)
    if anat_path and Path(anat_path).exists():
        sl = load_slice(Path(anat_path), slice_idx, axis, rot=rot)
        if bbox is not None:
            x0, x1, y0, y1 = bbox
            if x1 <= sl.shape[1] and y1 <= sl.shape[0]:
                sl = sl[y0:y1, x0:x1]
            else:
                print("[info] anatomical grid larger than crop bbox; "
                      "anat shown uncropped")
        ax.imshow(sl, cmap="gray", aspect="equal")
    else:
        ax.set_facecolor("black")


# =============================================================== #
#  Main composer
# =============================================================== #
def make_montage(
    data_dir, out_path,
    slice_idx=None, axis=2, rot=1,
    windows=None, window_overrides=None,
    auto_window=False, auto_pct=(0.5, 99.5),
    anatomical=None, mask=None,
    crop=None, auto_crop=False, crop_pad=2,
    dpi=300,
):
    """Compose the 3x4 montage and save it.

    ``mask`` may be a path to a NIfTI file. If ``None``, a file named
    ``Mask.nii.gz`` is searched for in ``data_dir`` and used if found.
    Pass the string "none" (case-insensitive) to disable masking
    entirely even if Mask.nii.gz is present.

    ``crop`` is an explicit (x0, x1, y0, y1) tuple in display-pixel
    coordinates. Takes precedence over ``auto_crop``. ``auto_crop``
    derives the same kind of bbox from the mask (with ``crop_pad``
    voxels of slack on each side) so the brain fills the panels.

    ``auto_window`` enables ImageJ-style percentile-based windowing,
    using ``auto_pct = (lo, hi)`` percentiles. Paired NLLS/NN maps
    pool their voxels so the resulting window is shared. With a mask,
    only in-mask voxels feed the percentile.

    ``window_overrides`` is a dict mapping param_id -> (vmin, vmax).
    These take precedence over both defaults and auto-window, so
    explicit user choices always win.
    """
    data_dir = Path(data_dir)
    windows = dict(windows) if windows else dict(WINDOWS)
    window_overrides = dict(window_overrides) if window_overrides else {}

    # ---- resolve mask -------------------------------------------- #
    mask_slc = None
    if isinstance(mask, str) and mask.lower() == "none":
        mask_path = None
    elif mask is None:
        candidate = data_dir / "Mask.nii.gz"
        mask_path = candidate if candidate.exists() else None
    else:
        mask_path = Path(mask)

    if mask_path is not None:
        if not mask_path.exists():
            raise FileNotFoundError(f"mask not found: {mask_path}")
        mask_slc = load_slice(mask_path, slice_idx, axis, rot=rot)
        print(f"[ok] mask loaded from {mask_path.name} "
              f"(shape {mask_slc.shape}, "
              f"{int((mask_slc > 0).sum())} voxels inside)")
    else:
        print("[info] no mask applied")

    # ---- resolve crop bbox -------------------------------------- #
    if crop is not None:
        bbox = tuple(crop)
        print(f"[ok] manual crop: x={bbox[0]}:{bbox[1]} y={bbox[2]}:{bbox[3]}")
    elif auto_crop:
        if mask_slc is None:
            raise ValueError("--auto-crop requires a mask")
        bbox = compute_mask_bbox(mask_slc, pad=crop_pad)
        if bbox is None:
            print("[warn] mask is empty; skipping auto-crop")
        else:
            print(f"[ok] auto-crop bbox (x0,x1,y0,y1) = {bbox} "
                  f"(pad={crop_pad}); pass --crop {bbox[0]},{bbox[1]},"
                  f"{bbox[2]},{bbox[3]} to reproduce")
    else:
        bbox = None

    # Crop the mask once so apply_mask shape-checks line up downstream.
    mask_slc = crop_slice(mask_slc, bbox) if mask_slc is not None else None

    # ---- pre-load parametric slices (raw, cropped, NOT yet masked) #
    # Done up-front so auto-window sees the same data as the renderer.
    loaded = {}  # filename -> 2D ndarray
    for filename in FILE_MAP:
        f = data_dir / filename
        if f.exists():
            slc = load_slice(f, slice_idx, axis, rot=rot)
            loaded[filename] = crop_slice(slc, bbox)

    # ---- auto-window -------------------------------------------- #
    if auto_window:
        # Group filenames by param_id so paired NLLS/NN maps share a
        # window computed from their pooled voxels.
        by_param = {}
        for filename, (param_id, _, _) in FILE_MAP.items():
            if filename in loaded:
                by_param.setdefault(param_id, []).append(loaded[filename])

        in_mask = (mask_slc > 0) if mask_slc is not None else None
        for param_id, slices in by_param.items():
            if param_id in window_overrides:
                continue  # explicit user override always wins
            if in_mask is not None:
                pooled = np.concatenate([s[in_mask].ravel() for s in slices])
            else:
                pooled = np.concatenate([s.ravel() for s in slices])
            win = compute_auto_window(pooled, *auto_pct)
            if win is not None:
                windows[param_id] = win
                print(f"[ok] auto-window {param_id}: "
                      f"vmin={win[0]:.4g} vmax={win[1]:.4g} "
                      f"(percentiles {auto_pct[0]}-{auto_pct[1]})")

    # Explicit -w overrides applied last so they always win.
    windows.update(window_overrides)

    # ---- figure -------------------------------------------------- #
    fig = plt.figure(figsize=(12, 9), facecolor="white")
    gs = GridSpec(
        3, 4, figure=fig,
        hspace=0.18, wspace=0.30,
        left=0.02, right=0.98, top=0.97, bottom=0.04,
    )

    rows = [
        ["__ANAT__",
         "AUC_pyr.nii.gz", "AUC_lac.nii.gz", "AUC_ratio.nii.gz"],
        ["(B) NLLS",
         "vB_NLLS_map_pair01.nii.gz",
         "kVE_NLLS_map_pair01.nii.gz",
         "kPL_NLLS_map_pair01.nii.gz"],
        ["(C) NN",
         "vB_map_NN_pair01.nii.gz",
         "kVE_map_NN_pair01.nii.gz",
         "kPL_map_NN_pair01.nii.gz"],
    ]

    for r, row in enumerate(rows):
        for c, key in enumerate(row):
            ax = fig.add_subplot(gs[r, c])

            if key == "__ANAT__":
                _draw_anat(ax, anatomical, slice_idx, axis, rot, bbox=bbox)
                continue

            if key.startswith("("):
                _draw_label_cell(ax, key)
                continue

            if key not in loaded:
                _strip_axes(ax)
                ax.set_facecolor("black")
                print(f"[warn] missing {key}; cell left blank")
                continue

            param, label, units = FILE_MAP[key]
            vmin, vmax = windows[param]
            slc = apply_mask(loaded[key], mask_slc)
            _draw_panel(ax, fig, slc, vmin, vmax, label, units)

    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


# =============================================================== #
#  CLI
# =============================================================== #
def _parse_window(s):
    """Parse 'PARAM=LO,HI' override string -> ('PARAM', (lo, hi))."""
    try:
        name, vals = s.split("=")
        lo, hi = vals.split(",")
        return name.strip(), (float(lo), float(hi))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"bad --window value '{s}'; expected PARAM=LO,HI"
        )


def _parse_crop(s):
    """Parse 'X0,X1,Y0,Y1' -> tuple of four ints."""
    try:
        parts = [int(p) for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError
        x0, x1, y0, y1 = parts
        if x1 <= x0 or y1 <= y0:
            raise ValueError("X1 must be > X0 and Y1 must be > Y0")
        return (x0, x1, y0, y1)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"bad --crop value '{s}'; expected X0,X1,Y0,Y1 ({e})"
        )


def _parse_pct(s):
    """Parse 'LO,HI' percentile pair -> (lo, hi) floats in [0, 100]."""
    try:
        lo, hi = (float(p) for p in s.split(","))
        if not (0 <= lo < hi <= 100):
            raise ValueError("require 0 <= LO < HI <= 100")
        return (lo, hi)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"bad --auto-window-pct value '{s}'; expected LO,HI ({e})"
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("data_dir", type=Path,
                    help="Folder containing the NIfTI maps")
    ap.add_argument("-o", "--out", type=Path, default=Path("montage.png"),
                    help="Output figure path (default: montage.png). "
                         "Extension determines format -- .png, .pdf, "
                         ".svg, .tiff, .eps all supported.")
    ap.add_argument("--slice", type=int, default=None,
                    help="Slice index along --axis (default: middle)")
    ap.add_argument("--axis", type=int, default=2, choices=[0, 1, 2],
                    help="Slice axis (default: 2)")
    ap.add_argument("--rot", type=int, default=1,
                    help="Number of 90 deg CCW rotations applied to "
                         "each 2D slice for display (default: 1)")
    ap.add_argument("--anat", type=Path, default=None,
                    help="Optional anatomical NIfTI for the top-left panel")
    ap.add_argument("--mask", type=Path, default=None,
                    help="Mask NIfTI applied to every metabolic/AUC "
                         "panel (not the anatomical). If omitted, the "
                         "script auto-detects Mask.nii.gz in DATA_DIR.")
    ap.add_argument("--no-mask", action="store_true",
                    help="Disable masking even if Mask.nii.gz is present")
    ap.add_argument("--auto-crop", action="store_true",
                    help="Crop every panel to the mask's bounding box "
                         "(plus --crop-pad voxels) so the brain fills "
                         "the panels. Requires a mask.")
    ap.add_argument("--crop-pad", type=int, default=2,
                    help="Voxels of padding around the auto-crop bbox "
                         "(default: 2)")
    ap.add_argument("--crop", type=_parse_crop, default=None,
                    metavar="X0,X1,Y0,Y1",
                    help="Manual crop in display-pixel coords (origin "
                         "top-left, X1/Y1 exclusive). Overrides "
                         "--auto-crop. Example: --crop 8,40,5,28")
    ap.add_argument("--auto-window", action="store_true",
                    help="ImageJ-style automatic window/level. "
                         "Pools voxels across paired NLLS/NN maps so "
                         "the pair shares one window. Uses in-mask "
                         "voxels when a mask is present. Explicit -w "
                         "overrides still win.")
    ap.add_argument("--auto-window-pct", type=_parse_pct,
                    default=(0.5, 99.5), metavar="LO,HI",
                    help="Percentile bounds for --auto-window "
                         "(default: 0.5,99.5; ~1%% saturated total). "
                         "Tighter values (e.g. 2,98) give more "
                         "aggressive contrast.")
    ap.add_argument("--dpi", type=int, default=300,
                    help="Output DPI (default: 300)")
    ap.add_argument("-w", "--window", action="append", default=[],
                    metavar="PARAM=LO,HI", type=_parse_window,
                    help="Override window/level, e.g. -w kPL=0,0.04 "
                         "(repeatable). PARAM in {AUC_pyr, AUC_lac, "
                         "AUC_ratio, vB, kVE, kPL}.")
    args = ap.parse_args()

    # Explicit -w values are passed as overrides so they beat auto-window.
    overrides = {}
    for name, vals in args.window:
        if name not in WINDOWS:
            ap.error(f"unknown parameter '{name}' in --window "
                     f"(choose from {sorted(WINDOWS)})")
        overrides[name] = vals

    mask_arg = "none" if args.no_mask else args.mask

    make_montage(
        args.data_dir, args.out,
        slice_idx=args.slice, axis=args.axis, rot=args.rot,
        windows=WINDOWS, window_overrides=overrides,
        auto_window=args.auto_window, auto_pct=args.auto_window_pct,
        anatomical=args.anat,
        mask=mask_arg, crop=args.crop, auto_crop=args.auto_crop,
        crop_pad=args.crop_pad, dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
