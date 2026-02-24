import numpy as np
from scipy.optimize import curve_fit

# Reuse measured-driver primitives to keep model parity with the generator
# from two_compartment_generator import (
#     _piecewise_app_R1,
#     _simulate_Pe_from_Sp,
#     _simulate_L_from_Pe,
# )

# ================================
# Measured‑Pyruvate Driver 
# ================================
def _rf_loss_per_TR(theta_deg, TR):
    theta = np.deg2rad(theta_deg)
    return -np.log(np.clip(np.cos(theta), 1e-8, 1.0)) / float(TR)


def _piecewise_app_R1(time_points, flips_deg, TR, R1_base):
    flips_deg = np.atleast_1d(np.array(flips_deg, dtype=float)).ravel()
    if flips_deg.size == 1:
        flips_deg = np.repeat(flips_deg, time_points.size)
    rf_losses = _rf_loss_per_TR(flips_deg[:time_points.size], TR)
    return R1_base + rf_losses
def _simulate_Pe_from_Sp(t, S_pyr, kpl, kve, vb, R1app_P):
    """Forward integration for intracellular pyruvate using measured total pyruvate signal.
    dPe/dt = (kve/vb)*S_pyr - [ (kve*(1 - vb)/vb) + kpl + R1app_P(t) ] * Pe
    """
    t = np.asarray(t, float)
    Sp = np.asarray(S_pyr, float)
    R1app_P = np.asarray(R1app_P, float)
    dt = np.diff(t, prepend=t[0])
    Pe = np.zeros_like(Sp, dtype=float)
    gain = (kve / np.clip(vb, 1e-6, 1.0))
    for k in range(1, t.size):
        loss = (kve * (1.0 - vb) / np.clip(vb, 1e-6, 1.0)) + kpl + R1app_P[k-1]
        dPe = gain * Sp[k-1] - loss * Pe[k-1]
        Pe[k] = Pe[k-1] + dt[k] * dPe
    return Pe
def _simulate_L_from_Pe(t, Pe, kpl, R1app_L):
    """dL/dt = kpl * Pe - R1app_L(t) * L"""
    t = np.asarray(t, float)
    Pe = np.asarray(Pe, float)
    R1app_L = np.asarray(R1app_L, float)
    dt = np.diff(t, prepend=t[0])
    L = np.zeros_like(Pe, dtype=float)
    for k in range(1, t.size):
        L[k] = L[k-1] + dt[k] * (kpl * Pe[k-1] - R1app_L[k-1] * L[k-1])
    return L


def _predict_lac_from_measured_pyr(
    t,
    S_pyr,
    kpl,
    kve,
    vb,
    R1p,
    R1l,
    theta_pyr_deg,
    theta_lac_deg,
    TR,
):
    """
    Forward model for S_lac(t) given measured S_pyr(t) as the driver.

    Parameters
    ----------
    t : (T,) array
    S_pyr : (T,) array      measured pyruvate signal (driver)
    kpl, kve, vb : floats   parameters to estimate
    R1p, R1l : floats       fixed apparent T1^-1 baselines (sec^-1)
    theta_pyr_deg, theta_lac_deg : (T,) array or scalar flip angles (deg)
    TR : float              repetition time (sec)

    Returns
    -------
    S_lac_pred : (T,) array
    Pe : (T,) array         intracellular pyruvate (for inspection)
    L : (T,) array          lactate pool (for inspection)
    """
    # Effective (piecewise) relaxation including RF loss, matching your simulator
    R1app_P = _piecewise_app_R1(t, theta_pyr_deg, TR, R1p)
    R1app_L = _piecewise_app_R1(t, theta_lac_deg, TR, R1l)

    # 1) integrate Pe from measured S_pyr
    Pe = _simulate_Pe_from_Sp(t, S_pyr, kpl, kve, vb, R1app_P)

    # 2) integrate L from Pe
    L = _simulate_L_from_Pe(t, Pe, kpl, R1app_L)

    # 3) measured lactate readout (extravascular fraction)
    S_lac_pred = (1.0 - vb) * L
    return S_lac_pred, Pe, L


# Uses measured S_pyr as a driver, solves the Pe/L ODEs with the 
# same primitives as your simulator, and fits only kpl, kve, and vb 
# to match measured S_lac (returns Pe, L, and S_lac_pred).
def fit_measured_driver_curvefit(
    time,
    S_pyr,
    S_lac,
    theta_pyr_deg,
    theta_lac_deg,
    TR=2.0,
    R1p=1/30.0,
    R1l=1/25.0,
    x0=None,
    bounds=None,
    sigma=None,
    absolute_sigma=False,
):
    """
    Curve-fit (kpl, kve, vb) using measured S_pyr as the driver.

    Parameters
    ----------
    time : (T,) array
    S_pyr : (T,) array
    S_lac : (T,) array
    theta_pyr_deg, theta_lac_deg : scalar or (T,) arrays of flip angles (deg)
                                   (can pass your VFA schedules here)
    TR : float
    R1p, R1l : floats
    x0 : initial guess [kpl, kve, vb]
         default = [0.05, 0.20, 0.10]
    bounds : ((low_kpl, low_kve, low_vb), (high_kpl, high_kve, high_vb))
             default = ([0.001, 0.01, 0.005], [0.5, 1.0, 0.95])
    sigma : (T,) array of per-point std devs for S_lac (optional weighting)
    absolute_sigma : passed to curve_fit

    Returns
    -------
    result : dict with keys
        'kpl', 'kve', 'vb'          fitted params
        'pcov'                      covariance matrix from curve_fit
        'S_lac_pred'                fitted lactate signal
        'Pe', 'L'                   state trajectories
        'success'                   True if convergence
        'message'                   optimizer message
    """
    t = np.asarray(time, dtype=float).ravel()
    Sp = np.asarray(S_pyr, dtype=float).ravel()
    Sl = np.asarray(S_lac, dtype=float).ravel()

    # Broadcast scalar flip angles to time grid if needed
    def _as_sched(th):
        arr = np.atleast_1d(np.array(th, dtype=float)).ravel()
        return arr if arr.size == t.size else np.full_like(t, float(arr[0]))
    thp = _as_sched(theta_pyr_deg)
    thl = _as_sched(theta_lac_deg)

    # Defaults
    if x0 is None:
        x0 = np.array([0.05, 0.20, 0.10], dtype=float)  # kpl, kve, vb
    if bounds is None:
        bounds = (
            np.array([0.001, 0.01, 0.005], dtype=float),
            np.array([0.50,  1.00, 0.95 ], dtype=float),
        )

    # Model wrapper for curve_fit: return S_lac_pred given params
    def _model_for_curvefit(t_local, kpl, kve, vb):
        S_lac_pred, _, _ = _predict_lac_from_measured_pyr(
            t_local, Sp, kpl, kve, vb, R1p, R1l, thp, thl, TR
        )
        return S_lac_pred

    # Fit (we only pass time; other data captured via closure)
    popt, pcov = curve_fit(
        _model_for_curvefit,
        t,
        Sl,
        p0=x0,
        bounds=bounds,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        maxfev=20000,
    )

    kpl_hat, kve_hat, vb_hat = popt.tolist()
    S_lac_pred, Pe, L = _predict_lac_from_measured_pyr(
        t, Sp, kpl_hat, kve_hat, vb_hat, R1p, R1l, thp, thl, TR
    )

    return {
        "kpl": float(kpl_hat),
        "kve": float(kve_hat),
        "vb":  float(vb_hat),
        "pcov": pcov,
        "S_lac_pred": S_lac_pred,
        "Pe": Pe,
        "L": L,
        "success": True,
        "message": "Converged",
    }
