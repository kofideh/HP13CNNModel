# fit_two_compartment_physio_nlls.py

import numpy as np
from scipy.optimize import least_squares


# ============================================================
# Utilities
# ============================================================

def _as_1d(x, name="array"):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError(f"{name} is empty")
    return x



def _rf_loss_per_TR(theta_deg, TR):
    theta_deg = np.minimum(np.asarray(theta_deg, dtype=float), 89.9)
    theta_rad = np.deg2rad(theta_deg)
    cosv = np.cos(theta_rad)
    cosv = np.clip(cosv, 1e-8, 1.0)
    return -np.log(cosv) / float(TR)


def _piecewise_app_R1(time_points, flips_deg, TR, R1_base):
    """
    Construct piecewise apparent R1:
        R1app(t_k) = R1_base - ln(cos(theta_k)) / TR
    """
    t = _as_1d(time_points, "time_points")
    flips = np.asarray(flips_deg, dtype=float).reshape(-1)

    if flips.size == 1:
        flips = np.repeat(flips, t.size)

    if flips.size < t.size:
        raise ValueError(
            "flip-angle schedule must be scalar or have at least len(time_points) entries"
        )

    flips = np.minimum(flips[:t.size], 89.9)
    return float(R1_base) + _rf_loss_per_TR(flips, TR)


def _exp_update_linear(dt, a, b, x_prev):
    """
    Exact update for:
        dx/dt = a - b*x
    assuming a and b are constant over one time step.
    """
    b = float(b)

    if b <= 1e-12:
        return x_prev + dt * (a - b * x_prev)

    e = np.exp(-b * dt)
    return x_prev * e + (a / b) * (1.0 - e)


def _default_channel_scales(S_pyr, S_lac, sigma_pyr=None, sigma_lac=None):
    """
    If known noise standard deviations are available, pass them explicitly.
    Otherwise, this defaults to peak-normalized residuals so that pyruvate
    does not completely dominate lactate because of its larger amplitude.
    """
    if sigma_pyr is None:
        sigma_pyr = max(np.nanmax(np.abs(S_pyr)), 1e-12)

    if sigma_lac is None:
        sigma_lac = max(np.nanmax(np.abs(S_lac)), 1e-12)

    return float(sigma_pyr), float(sigma_lac)


def _clip_initial(p0, lb, ub):
    p0 = np.asarray(p0, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    eps = 1e-10 * (ub - lb + 1.0)
    return np.minimum(np.maximum(p0, lb + eps), ub - eps)


# ============================================================
# Corrected physiologic forward model
# ============================================================

def gamma_variate_input(time_points, peak_amp, alpha, beta, t0):
    """
    Draw or fit a vascular pyruvate input Pv(t), represented as a
    gamma-variate-like bolus curve.

    This is NOT the measured voxel pyruvate signal S_pyr(t).
    The measured pyruvate signal is generated downstream as:

        S_pyr(t) = vb * Pv(t) + (1 - vb) * Pe(t)
    """
    t = _as_1d(time_points, "time_points")
    tau = np.maximum(t - float(t0), 0.0)

    raw = (tau ** float(alpha)) * np.exp(-float(beta) * tau)
    peak = np.max(raw)

    if peak <= 1e-12:
        return np.zeros_like(t)

    Pv = float(peak_amp) * raw / peak
    return np.clip(Pv, 0.0, None)


def simulate_physio_from_Pv(
    time_points,
    Pv,
    kpl,
    kve,
    vb,
    R1app_P,
    R1app_L,
):
    """
    Corrected two-compartment model:

        dPe/dt = kve * Pv(t) - [kve + kpl + R1Papp(t)] * Pe(t)

        dL/dt  = kpl * Pe(t) - R1Lapp(t) * L(t)

        S_pyr(t) = vb * Pv(t) + (1 - vb) * Pe(t)

        S_lac(t) = (1 - vb) * L(t)

    Parameters
    ----------
    Pv : array
        Vascular pyruvate input function, not the measured pyruvate signal.
    """

    t = _as_1d(time_points, "time_points")
    Pv = _as_1d(Pv, "Pv")
    R1app_P = _as_1d(R1app_P, "R1app_P")
    R1app_L = _as_1d(R1app_L, "R1app_L")

    if Pv.size != t.size:
        raise ValueError("Pv and time_points must have the same length")

    if R1app_P.size != t.size or R1app_L.size != t.size:
        raise ValueError("R1app_P/R1app_L and time_points must have the same length")

    kpl = float(kpl)
    kve = float(kve)
    vb = float(vb)

    dt = np.diff(t, prepend=t[0])

    Pe = np.zeros_like(Pv, dtype=float)
    L = np.zeros_like(Pv, dtype=float)

    for k in range(1, t.size):
        # Pe' = kve*Pv - (kve + kpl + R1Papp)*Pe
        a_pe = kve * Pv[k - 1]
        b_pe = kve + kpl + R1app_P[k - 1]
        Pe[k] = _exp_update_linear(dt[k], a_pe, b_pe, Pe[k - 1])

        # L' = kpl*Pe - R1Lapp*L
        a_l = kpl * Pe[k - 1]
        b_l = R1app_L[k - 1]
        L[k] = _exp_update_linear(dt[k], a_l, b_l, L[k - 1])

    S_pyr = vb * Pv + (1.0 - vb) * Pe
    S_lac = (1.0 - vb) * L

    return S_pyr, S_lac, Pe, L


# ============================================================
# NLLS option 1: oracle vascular input Pv is known
# ============================================================

def fit_nlls_known_Pv(
    time_points,
    Pv,
    S_pyr_obs,
    S_lac_obs,
    TR=2.0,
    flips_pyr_deg=11.0,
    flips_lac_deg=80.0,
    R1p=1 / 30.0,
    R1l=1 / 25.0,
    p0=(0.05, 0.15, 0.09),
    bounds=((1e-4, 1e-3, 0.01), (1.0, 1.0, 0.5)),
    sigma_pyr=None,
    sigma_lac=None,
    max_nfev=40000,
):
    """
    NLLS with known vascular input Pv(t).

    This is the best first sanity check for synthetic data.
    If the corrected model is implemented properly, this should recover
    kpl, kve, and vb well at high SNR.
    """

    t = _as_1d(time_points, "time_points")
    Pv = _as_1d(Pv, "Pv")
    S_pyr_obs = _as_1d(S_pyr_obs, "S_pyr_obs")
    S_lac_obs = _as_1d(S_lac_obs, "S_lac_obs")

    if not (Pv.size == S_pyr_obs.size == S_lac_obs.size == t.size):
        raise ValueError(
            "time_points, Pv, S_pyr_obs, and S_lac_obs must have the same length"
        )

    R1app_P = _piecewise_app_R1(t, flips_pyr_deg, TR, R1p)
    R1app_L = _piecewise_app_R1(t, flips_lac_deg, TR, R1l)

    sigma_pyr, sigma_lac = _default_channel_scales(
        S_pyr_obs, S_lac_obs, sigma_pyr, sigma_lac
    )

    lb, ub = [np.asarray(x, dtype=float) for x in bounds]
    p0 = _clip_initial(p0, lb, ub)

    def residual(theta):
        kpl, kve, vb = theta

        S_pyr_pred, S_lac_pred, _, _ = simulate_physio_from_Pv(
            t, Pv, kpl, kve, vb, R1app_P, R1app_L
        )

        r_pyr = (S_pyr_pred - S_pyr_obs) / sigma_pyr
        r_lac = (S_lac_pred - S_lac_obs) / sigma_lac

        return np.concatenate([r_pyr, r_lac])

    res = least_squares(
        residual,
        p0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=max_nfev,
        x_scale="jac",
    )

    kpl, kve, vb = res.x

    S_pyr_fit, S_lac_fit, Pe_fit, L_fit = simulate_physio_from_Pv(
        t, Pv, kpl, kve, vb, R1app_P, R1app_L
    )

    return {
        "success": bool(res.success),
        "message": res.message,
        "cost": float(res.cost),
        "optimality": float(res.optimality),
        "kpl": float(kpl),
        "kve": float(kve),
        "vb": float(vb),
        "Pv_fit": Pv.copy(),
        "Pe_fit": Pe_fit,
        "L_fit": L_fit,
        "S_pyr_fit": S_pyr_fit,
        "S_lac_fit": S_lac_fit,
        "result": res,
    }


# ============================================================
# NLLS option 2: fit gamma-variate vascular input Pv(t)
# ============================================================

def fit_nlls_gamma_AIF(
    time_points,
    S_pyr_obs,
    S_lac_obs,
    TR=2.0,
    flips_pyr_deg=11.0,
    flips_lac_deg=80.0,
    R1p=1 / 30.0,
    R1l=1 / 25.0,
    p0=None,
    bounds=None,
    sigma_pyr=None,
    sigma_lac=None,
    n_starts=10,
    seed=42,
    max_nfev=40000,
):
    """
    NLLS where the vascular pyruvate input Pv(t) is modeled as a
    gamma-variate function and fitted jointly with kpl, kve, and vb.

    Parameter vector:
        theta = [kpl, kve, vb, AIF_peak_amp, AIF_alpha, AIF_beta, AIF_t0]

    This is closer to a realistic NLLS benchmark when Pv(t) is not known.
    """

    t = _as_1d(time_points, "time_points")
    S_pyr_obs = _as_1d(S_pyr_obs, "S_pyr_obs")
    S_lac_obs = _as_1d(S_lac_obs, "S_lac_obs")

    if not (S_pyr_obs.size == S_lac_obs.size == t.size):
        raise ValueError("time_points, S_pyr_obs, and S_lac_obs must have the same length")

    R1app_P = _piecewise_app_R1(t, flips_pyr_deg, TR, R1p)
    R1app_L = _piecewise_app_R1(t, flips_lac_deg, TR, R1l)

    sigma_pyr, sigma_lac = _default_channel_scales(
        S_pyr_obs, S_lac_obs, sigma_pyr, sigma_lac
    )

    peak_obs = max(float(np.nanmax(S_pyr_obs)), 1e-6)
    t_peak = float(t[int(np.nanargmax(S_pyr_obs))])

    default_aif_lb = np.array([
        0.05 * peak_obs,          # AIF peak amplitude
        1.0,                      # alpha
        0.05,                     # beta
        t[0] - 2 * TR,            # t0
    ], dtype=float)

    default_aif_ub = np.array([
        20.0 * peak_obs,          # AIF peak amplitude
        6.0,                      # alpha
        2.0,                      # beta
        min(t[-1], t_peak),       # t0
    ], dtype=float)

    default_aif_p0 = np.array([
        peak_obs,
        3.0,
        0.8,
        max(t[0] - TR, t_peak - 4.0),
    ], dtype=float)

    if bounds is None:
        kinetic_lb = np.array([1e-4, 1e-3, 0.01], dtype=float)
        kinetic_ub = np.array([1.0, 1.0, 0.5], dtype=float)
        lb = np.concatenate([kinetic_lb, default_aif_lb])
        ub = np.concatenate([kinetic_ub, default_aif_ub])
    else:
        kinetic_or_full_lb, kinetic_or_full_ub = [np.asarray(x, dtype=float) for x in bounds]

        if kinetic_or_full_lb.size == 3:
            lb = np.concatenate([kinetic_or_full_lb, default_aif_lb])
            ub = np.concatenate([kinetic_or_full_ub, default_aif_ub])
        elif kinetic_or_full_lb.size == 7:
            lb = kinetic_or_full_lb
            ub = kinetic_or_full_ub
        else:
            raise ValueError(
                "bounds must contain either 3 kinetic parameters "
                "[kPL, kVE, vB] or 7 parameters "
                "[kPL, kVE, vB, AIF_peak_amp, AIF_alpha, AIF_beta, AIF_t0]"
            )

    if p0 is None:
        kinetic_p0 = np.array([0.05, 0.15, 0.09], dtype=float)
        p0 = np.concatenate([kinetic_p0, default_aif_p0])
    else:
        p0 = np.asarray(p0, dtype=float)

        if p0.size == 3:
            p0 = np.concatenate([p0, default_aif_p0])
        elif p0.size != 7:
            raise ValueError(
                "p0 must contain either 3 kinetic parameters "
                "[kPL, kVE, vB] or 7 parameters "
                "[kPL, kVE, vB, AIF_peak_amp, AIF_alpha, AIF_beta, AIF_t0]"
            )

    p0 = _clip_initial(p0, lb, ub)

    def residual(theta):
        kpl, kve, vb, peak_amp, alpha, beta, t0 = theta

        Pv = gamma_variate_input(t, peak_amp, alpha, beta, t0)

        S_pyr_pred, S_lac_pred, _, _ = simulate_physio_from_Pv(
            t, Pv, kpl, kve, vb, R1app_P, R1app_L
        )

        r_pyr = (S_pyr_pred - S_pyr_obs) / sigma_pyr
        r_lac = (S_lac_pred - S_lac_obs) / sigma_lac

        return np.concatenate([r_pyr, r_lac])

    rng = np.random.RandomState(seed)

    starts = [p0]

    for _ in range(max(0, int(n_starts) - 1)):
        x = lb + rng.rand(lb.size) * (ub - lb)

        # Put random starts in a physiologically plausible region rather than
        # uniformly sampling extreme corners of the full bounded space.
        x[0] = rng.uniform(0.005, min(0.25, ub[0]))          # kpl
        x[1] = rng.uniform(max(lb[1], 0.02), min(0.5, ub[1])) # kve
        x[2] = rng.uniform(max(lb[2], 0.02), min(0.20, ub[2])) # vb
        x[3] = rng.uniform(0.5 * peak_obs, 3.0 * peak_obs)  # AIF peak
        x[4] = rng.uniform(2.0, 4.5)                        # alpha
        x[5] = rng.uniform(0.2, 1.2)                        # beta
        x[6] = rng.uniform(lb[6], ub[6])                    # t0

        starts.append(_clip_initial(x, lb, ub))

    best = None

    for x0 in starts:
        res = least_squares(
            residual,
            x0,
            bounds=(lb, ub),
            method="trf",
            max_nfev=max_nfev,
            x_scale="jac",
        )

        if best is None or res.cost < best.cost:
            best = res

    kpl, kve, vb, peak_amp, alpha, beta, t0 = best.x

    Pv_fit = gamma_variate_input(t, peak_amp, alpha, beta, t0)

    S_pyr_fit, S_lac_fit, Pe_fit, L_fit = simulate_physio_from_Pv(
        t, Pv_fit, kpl, kve, vb, R1app_P, R1app_L
    )

    return {
        "success": bool(best.success),
        "message": best.message,
        "cost": float(best.cost),
        "optimality": float(best.optimality),
        "kpl": float(kpl),
        "kve": float(kve),
        "vb": float(vb),
        "AIF_peak_amp": float(peak_amp),
        "AIF_alpha": float(alpha),
        "AIF_beta": float(beta),
        "AIF_t0": float(t0),
        "Pv_fit": Pv_fit,
        "Pe_fit": Pe_fit,
        "L_fit": L_fit,
        "S_pyr_fit": S_pyr_fit,
        "S_lac_fit": S_lac_fit,
        "result": best,
    }
    
    
    
def fit_nlls_known_Pv_multistart(
    time_points,
    Pv,
    S_pyr_obs,
    S_lac_obs,
    TR=2.0,
    flips_pyr_deg=11.0,
    flips_lac_deg=80.0,
    R1p=1 / 30.0,
    R1l=1 / 25.0,
    bounds=((1e-4, 1e-3, 0.01), (1.0, 1.0, 0.5)),
    sigma_pyr=None,
    sigma_lac=None,
    n_starts=20,
    seed=42,
    max_nfev=40000,
):
    t = _as_1d(time_points, "time_points")
    Pv = _as_1d(Pv, "Pv")
    S_pyr_obs = _as_1d(S_pyr_obs, "S_pyr_obs")
    S_lac_obs = _as_1d(S_lac_obs, "S_lac_obs")

    R1app_P = _piecewise_app_R1(t, flips_pyr_deg, TR, R1p)
    R1app_L = _piecewise_app_R1(t, flips_lac_deg, TR, R1l)

    sigma_pyr, sigma_lac = _default_channel_scales(
        S_pyr_obs, S_lac_obs, sigma_pyr, sigma_lac
    )

    lb, ub = [np.asarray(x, dtype=float) for x in bounds]
    rng = np.random.RandomState(seed)

    def residual(theta):
        kpl, kve, vb = theta

        S_pyr_pred, S_lac_pred, _, _ = simulate_physio_from_Pv(
            t, Pv, kpl, kve, vb, R1app_P, R1app_L
        )

        r_pyr = (S_pyr_pred - S_pyr_obs) / sigma_pyr
        r_lac = (S_lac_pred - S_lac_obs) / sigma_lac

        return np.concatenate([r_pyr, r_lac])

    starts = []

    # midpoint start
    starts.append(0.5 * (lb + ub))

    # random starts
    for _ in range(n_starts - 1):
        starts.append(lb + rng.rand(3) * (ub - lb))

    best = None

    for x0 in starts:
        x0 = _clip_initial(x0, lb, ub)

        res = least_squares(
            residual,
            x0,
            bounds=(lb, ub),
            method="trf",
            max_nfev=max_nfev,
            x_scale="jac",
        )

        if best is None or res.cost < best.cost:
            best = res

    kpl, kve, vb = best.x

    S_pyr_fit, S_lac_fit, Pe_fit, L_fit = simulate_physio_from_Pv(
        t, Pv, kpl, kve, vb, R1app_P, R1app_L
    )

    return {
        "success": bool(best.success),
        "message": best.message,
        "cost": float(best.cost),
        "optimality": float(best.optimality),
        "kpl": float(kpl),
        "kve": float(kve),
        "vb": float(vb),
        "Pv_fit": Pv.copy(),
        "Pe_fit": Pe_fit,
        "L_fit": L_fit,
        "S_pyr_fit": S_pyr_fit,
        "S_lac_fit": S_lac_fit,
        "result": best,
    }