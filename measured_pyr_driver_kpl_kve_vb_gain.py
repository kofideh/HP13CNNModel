# fit_two_compartment_corrected.py
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import traceback

DEFAULT_VFA_SCHEDULE = np.array([
    14.4775, 14.9632, 15.5014, 16.1021, 16.7787, 17.5484, 18.4349, 19.4712,
    20.7048, 22.2077, 24.0948, 26.5651, 30.0000, 35.2644, 45.0000, 90.0000
], dtype=float)

DEFAULT_TIME_POINTS = np.arange(0, 32, 2) # 16 time points from 0 to 30s with TR=2s

def rf_loss(theta_deg, TR, method="logcos"):
    """RF-driven loss rate per second from a net flip over one volume TR."""
    theta = np.deg2rad(theta_deg)
    if method == "logcos":
        # exact: e^{-RF*TR} = cos(theta)  ->  RF = -ln(cos theta)/TR
        return -np.log(np.clip(np.cos(theta), 1e-8, 1.0)) / TR
    else:
        # small-angle (only for theta≲20°)
        return (1.0 - np.cos(theta)) / TR



def rf_losses_at_time(t, TR):
        """
        Return (rf_loss_p, rf_loss_l) at continuous time t by selecting the
        piecewise-constant TR bin based on self.time_points.
        """

        t0 = float(DEFAULT_TIME_POINTS[0])
        if t <= t0:
            idx = 0
        else:
            idx = int(np.floor((t - t0) / TR))
        idx = int(np.clip(idx, 0, len(DEFAULT_TIME_POINTS) - 1))
        
        rfp = rf_loss(DEFAULT_VFA_SCHEDULE[idx], TR)
        rfl = rf_loss(DEFAULT_VFA_SCHEDULE[idx], TR)
        return rfp, rfl


def fit_traditional_2state_model(time_points, S_pyr_noisy, S_lac_noisy, 
                                      estimate_r1=False, initial_guess=None, bounds=None,
                                      flip_angle_pyr_deg=11.0, flip_angle_lac_deg=80.0, TR=2.0):
    """
    Corrected fitting function that matches the vb_only generator model exactly.
    """
    # Don't normalize the input data - use it as-is to match generator
    combined_noisy_data = np.concatenate((S_pyr_noisy, S_lac_noisy))

    def _aif(t, t0=0, alpha=3.0, beta=1.0):
        """AIF function matching the generator"""
        t_shifted = np.maximum(t - t0, 0)
        return alpha * t_shifted * np.exp(-beta * t_shifted)

    def model_signal_wrapper(t, *params_to_fit):
        kpl, kve, vb = params_to_fit[0], params_to_fit[1], params_to_fit[2]
        
        # Use the same r1 values as the generator
        if estimate_r1:
            r1p, r1l = params_to_fit[3], params_to_fit[4]
        else:
            r1p = 1/30  # Match generator
            r1l = 1/25  # Match generator

        if not (0.001 < vb < 0.999):  # Allow wider range but prevent edge cases
            return np.ones_like(combined_noisy_data) * 1e10

        try:
            # Solve the 2-compartment ODE matching the generator exactly
            def deriv(y, t):
                # pyr_loss = r1p + rf_loss(flip_angle_pyr_deg, TR)
                # lac_loss = r1l + rf_loss(flip_angle_lac_deg, TR)
                pyr_loss = r1p + rf_losses_at_time(t, TR)[0]
                lac_loss = r1l + rf_losses_at_time(t, TR)[1]
                Pe, Le = y
                AIF = _aif(t, t0=0, alpha=3.0, beta=1.0)  # Match generator params
                dPe_dt = AIF - (kpl + kve + r1p + pyr_loss) * Pe
                dLe_dt = kpl * Pe - (r1l + lac_loss) * Le
                return [dPe_dt, dLe_dt]

            y0 = [0, 0]
            sol = odeint(deriv, y0, t)
            Pe, Le = sol[:, 0], sol[:, 1]
            
            # Calculate vascular component
            Pv = _aif(t, t0=0, alpha=3.0, beta=1.0)
            
            # Calculate signals exactly as in generator
            S_pyr = vb * Pv + (1 - vb) * Pe
            S_lac = (1 - vb) * Le
            
            # Don't normalize - return raw signals to match generator
            return np.concatenate((S_pyr, S_lac))
            
        except Exception as e:
            print(f"Model evaluation failed: {e}")
            return np.ones_like(combined_noisy_data) * 1e12

    # Set up parameter bounds and initial guess
    if estimate_r1:
        n_params_fit = 5
        default_guess = [0.05, 0.02, 0.09, 1/43, 1/33]  # Match generator defaults
        default_bounds = ([0.001, 0.001, 0.001, 0.01, 0.01], [0.5, 1.0, 0.999, 0.1, 0.1])
    else:
        n_params_fit = 3
        default_guess = [0.05, 0.02, 0.09]  # Match generator defaults
        default_bounds = ([0.001, 0.001, 0.001], [0.5, 1.0, 0.999])

    if initial_guess is None:
        initial_guess = default_guess
    if bounds is None:
        bounds = default_bounds

    try:
        fitted_params, covariance = curve_fit(
            model_signal_wrapper,
            time_points,
            combined_noisy_data,
            p0=initial_guess,
            bounds=bounds,
            method='trf',
            maxfev=2000 * n_params_fit  # Increase iterations
        )
        std_devs = np.sqrt(np.diag(covariance))
        success = True
    except Exception as e:
        print(f"Fit failed: {e}")
        fitted_params = np.array(initial_guess)
        std_devs = np.full_like(fitted_params, np.nan)
        success = False

    if not estimate_r1:
        full_params = np.concatenate((fitted_params, [1/43, 1/33]))  # Use generator r1 values
        full_std_devs = np.concatenate((std_devs, [0, 0]))
    else:
        full_params = fitted_params
        full_std_devs = std_devs

    return full_params, full_std_devs, success



# ...existing code...

def fit_traditional_3state_model(time_points, S_pyr_noisy, S_lac_noisy, 
                                      estimate_r1=False, initial_guess=None, bounds=None,
                                      flip_angle_pyr_deg=11.0, flip_angle_lac_deg=80.0, TR=2.0):
    """
    3-state model fitting function with explicit vascular pool:
    
    dPv/dt = Ktrans*AIF(t) - (kve + r1app_p)*Pv
    dPe/dt = kve*Pv - (kpl + r1app_p)*Pe  
    dLe/dt = kpl*Pe - r1app_l*Le
    
    Signals: S_pyr = vb*Pv + (1-vb)*Pe, S_lac = (1-vb)*Le
    """
    # Don't normalize the input data - use it as-is to match generator
    combined_noisy_data = np.concatenate((S_pyr_noisy, S_lac_noisy))

    def _aif(t, t0=0, alpha=3.0, beta=1.0):
        """AIF function matching the generator"""
        t_shifted = np.maximum(t - t0, 0)
        return alpha * t_shifted * np.exp(-beta * t_shifted)

    def model_signal_wrapper_3state(t, *params_to_fit):
        if estimate_r1:
            kpl, kve, vb, Ktrans, r1p, r1l = params_to_fit
        else:
            kpl, kve, vb, Ktrans = params_to_fit
            r1p = 1/43  # Match generator
            r1l = 1/33  # Match generator

        # Ensure vb is within valid range
        vb = np.clip(vb, 1e-6, 0.999)
        
        if not (0.001 < Ktrans < 2.0):  # Reasonable Ktrans range
            return np.ones_like(combined_noisy_data) * 1e10

        try:
            # Calculate apparent relaxation rates including RF losses
            r1app_p = r1p + rf_loss(flip_angle_pyr_deg, TR)
            r1app_l = r1l + rf_loss(flip_angle_lac_deg, TR)
            
            # Solve the 3-compartment ODE system
            def deriv_3state(y, t):
                Pv, Pe, Le = y
                AIF = _aif(t, t0=0, alpha=3.0, beta=1.0)  # Match generator params
                
                dPv_dt = Ktrans * AIF - (kve + r1app_p) * Pv
                dPe_dt = kve * Pv - (kpl + r1app_p) * Pe
                dLe_dt = kpl * Pe - r1app_l * Le
                
                return [dPv_dt, dPe_dt, dLe_dt]

            y0 = [0.0, 0.0, 0.0]  # Initial conditions: [Pv, Pe, Le]
            sol = odeint(deriv_3state, y0, t)
            Pv, Pe, Le = sol[:, 0], sol[:, 1], sol[:, 2]
            
            # Calculate signals as weighted sum of compartments
            S_pyr = vb * Pv + (1 - vb) * Pe
            S_lac = (1 - vb) * Le
            
            # Return concatenated signals
            return np.concatenate((S_pyr, S_lac))
            
        except Exception as e:
            print(f"3-state model evaluation failed: {e}")
            return np.ones_like(combined_noisy_data) * 1e12

    # Set up parameter bounds and initial guess for 3-state model
    if estimate_r1:
        n_params_fit = 6
        default_guess = [0.05, 0.2, 0.1, 0.2, 1/43, 1/33]  # [kpl, kve, vb, Ktrans, r1p, r1l]
        default_bounds = ([0.001, 0.001, 0.001, 0.001, 0.01, 0.01], 
                         [0.5, 1.0, 0.999, 2.0, 0.1, 0.1])
    else:
        n_params_fit = 4
        default_guess = [0.05, 0.2, 0.1, 0.2]  # [kpl, kve, vb, Ktrans]
        default_bounds = ([0.001, 0.001, 0.001, 0.001], 
                         [0.5, 1.0, 0.999, 2.0])

    if initial_guess is None:
        initial_guess = default_guess
    if bounds is None:
        bounds = default_bounds

    try:
        fitted_params, covariance = curve_fit(
            model_signal_wrapper_3state,
            time_points,
            combined_noisy_data,
            p0=initial_guess,
            bounds=bounds,
            method='trf',
            maxfev=3000 * n_params_fit  # More iterations for 3-state model
        )
        std_devs = np.sqrt(np.diag(covariance))
        success = True
    except Exception as e:
        print(f"3-state fit failed: {e}")
        fitted_params = np.array(initial_guess)
        std_devs = np.full_like(fitted_params, np.nan)
        success = False

    if not estimate_r1:
        # Return [kpl, kve, vb, Ktrans, r1p, r1l]
        full_params = np.concatenate((fitted_params, [1/43, 1/33]))
        full_std_devs = np.concatenate((std_devs, [0, 0]))
    else:
        full_params = fitted_params
        full_std_devs = std_devs

    return full_params, full_std_devs, success


def fit_traditional_2c_model(time_points, S_pyr_noisy, S_lac_noisy, 
                                          estimate_r1=False, initial_guess=None, bounds=None,
                                          flip_angle_pyr_deg=11.0, flip_angle_lac_deg=80.0, TR=2.0,
                                          use_3state=False):
    """
    Unified fitting function that can use either 2-state or 3-state model.
    
    Parameters:
    -----------
    use_3state : bool
        If True, uses 3-state model (Pv, Pe, Le)
        If False, uses 2-state model (Pe, Le) + vascular
    """
    if use_3state:
        return fit_traditional_3state_model(
            time_points, S_pyr_noisy, S_lac_noisy, estimate_r1, 
            initial_guess, bounds, flip_angle_pyr_deg, flip_angle_lac_deg, TR
        )
    else:
        return fit_traditional_2state_model(
            time_points, S_pyr_noisy, S_lac_noisy, estimate_r1, 
            initial_guess, bounds, flip_angle_pyr_deg, flip_angle_lac_deg, TR
        )

# ...existing code...
# =============================
# Measured-Pyruvate-Driver Fit
# =============================
def _rf_loss_per_TR(theta_deg, TR):
    theta = np.deg2rad(theta_deg)
    return -np.log(np.clip(np.cos(theta), 1e-8, 1.0)) / TR

def _piecewise_app_R1(time_points, flips_deg, TR, R1_base):
    flips_deg = np.asarray(flips_deg).reshape(-1)
    if len(flips_deg) == 1:
        flips_deg = np.repeat(flips_deg, len(time_points))
    rf_losses = _rf_loss_per_TR(flips_deg[:len(time_points)], TR)
    return R1_base + rf_losses

def _simulate_L_from_measured_P(t, P_drv, kpl, R1app_L):
    P_drv = np.asarray(P_drv).astype(float)
    R1app_L = np.asarray(R1app_L).astype(float)
    dt = np.diff(t, prepend=t[0])
    L = np.zeros_like(P_drv, dtype=float)
    for k in range(1, len(t)):
        L[k] = L[k-1] + dt[k] * (kpl * P_drv[k-1] - R1app_L[k-1] * L[k-1])
    return L

# =========================================================
# Measured-P driver with kve and vb estimation (no AIF)
# =========================================================
def _simulate_Pe_from_Sp(t, S_pyr, kpl, kve, vb, R1app_P):
    """
    Solve for intracellular pyruvate Pe(t) using measured S_pyr(t) and
    the relation S_pyr = vb*Pv + (1-vb)*Pe.
    Using Pv = (S_pyr - (1-vb)*Pe)/vb, the Pe ODE becomes:
        dPe/dt = (kve/vb)*S_pyr - [ (kve*(1 - vb)/vb) + kpl + R1app_P(t) ] * Pe
    """
    t = np.asarray(t).astype(float)
    S_pyr = np.asarray(S_pyr).astype(float)
    R1app_P = np.asarray(R1app_P).astype(float)
    dt = np.diff(t, prepend=t[0])
    Pe = np.zeros_like(S_pyr, dtype=float)
    gain = (kve / np.clip(vb, 1e-6, 1.0))
    for k in range(1, len(t)):
        loss = (kve * (1.0 - vb) / np.clip(vb, 1e-6, 1.0)) + kpl + R1app_P[k-1]
        dPe = gain * S_pyr[k-1] - loss * Pe[k-1]
        Pe[k] = Pe[k-1] + dt[k] * dPe
    return Pe

# Uses measured S_pyr but first reconstructs Pe from S_pyr, 
# then fits kpl, kve, vb plus a lactate gain term gL (and optionally R1l), 
# with optional smoothing and tighter bounds—no explicit AIF term.
def fit_measured_pyr_driver_kve_vb(time_points, S_pyr, S_lac,
                                   TR=2.0,
                                   flips_pyr_deg=11.0,
                                   flips_lac_deg=80.0,
                                   R1p=1/30.0,
                                   R1l=1/25.0,
                                   estimate_R1l=False,
                                   smooth_p_driver=False,
                                   vb_bounds=(0.01, 0.5),
                                   kve_bounds=(0.001, 1.0)):
    """Estimate [kPL, kVE, v_b, gL] (+ optional R1l) with measured pyruvate.
    Lactate model uses Pe(t) reconstructed from S_pyr(t).
    """
    t = np.asarray(time_points, float).reshape(-1)
    Sp = np.asarray(S_pyr, float).reshape(-1)
    Sl = np.asarray(S_lac, float).reshape(-1)

    if smooth_p_driver:
        lam = 0.2
        for k in range(1, len(Sp)):
            Sp[k] = (1 - lam) * Sp[k] + lam * Sp[k-1]

    # Apparent R1 including RF losses (per TR)
    R1app_P = _piecewise_app_R1(t, flips_pyr_deg, TR, R1p)
    R1app_L = _piecewise_app_R1(t, flips_lac_deg, TR, R1l)

    # Parameter vector: [kpl, kve, vb, gL] (+ R1l if estimate_R1l)
    # p0 = [0.05, 0.15, 0.10, np.max(Sl)/(np.max(Sp)+1e-12)]
    p0 = [0.05, 0.02, 0.09, np.max(Sl)/(np.max(Sp)+1e-12)]
    lb = [1e-4, kve_bounds[0], vb_bounds[0], 1e-6]
    ub = [1.0,  kve_bounds[1], vb_bounds[1], 1e3]

    if estimate_R1l:
        p0 += [R1l]
        lb += [1/60.0]
        ub += [1/10.0]

    def model(t_samples, *theta):
        if estimate_R1l:
            kpl, kve, vb, gL, R1l_est = theta
            R1app_L_use = _piecewise_app_R1(t_samples, flips_lac_deg, TR, R1l_est)
        else:
            kpl, kve, vb, gL = theta
            R1app_L_use = R1app_L
        # Reconstruct Pe(t) from measured S_pyr
        Pe = _simulate_Pe_from_Sp(t_samples, Sp, kpl, kve, vb, R1app_P)
        # Lactate from Pe(t)
        L = _simulate_L_from_measured_P(t_samples, Pe, kpl, R1app_L_use)
        return gL * L

    from scipy.optimize import curve_fit
    try:
        pars, cov = curve_fit(model, t, Sl, p0=p0, bounds=(lb, ub), method="trf", maxfev=40000)
        success = True
    except Exception as e:
        print("fit_measured_pyr_driver_kve_vb failed:", e)
        pars = np.array(p0, dtype=float)
        cov = np.full((len(p0), len(p0)), np.nan)
        success = False

    out = {"success": bool(success), "covariance": cov}
    if estimate_R1l:
        kpl, kve, vb, gL, R1l_out = pars
    else:
        kpl, kve, vb, gL = pars
        R1l_out = R1l

    Pe_fit = _simulate_Pe_from_Sp(t, Sp, kpl, kve, vb, R1app_P)
    L_fit = _simulate_L_from_measured_P(t, Pe_fit, kpl, _piecewise_app_R1(t, flips_lac_deg, TR, R1l_out))

    out.update({
        "kpl": float(kpl),
        "kve": float(kve),
        "vb": float(vb),
        "gL": float(gL),
        "R1l": float(R1l_out),
        "Pe_fit": Pe_fit,
        "S_lac_fit": gL * L_fit,
    })
    return out
