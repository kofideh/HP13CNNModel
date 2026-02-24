import numpy as np
from scipy.integrate import odeint


from hybrid_model_utils import VB_MIN, VB_MAX, KPL_MIN, KPL_MAX, KVE_MIN, KVE_MAX


DEFAULT_TR = 2.0  # seconds
DEFAULT_NUM_TIMEPOINTS = 25
# This import can stay (used elsewhere in your codebase), but we won't rely on it here.
# from fit_two_compartment import rf_loss  


DEFAULT_PYR_FA_SCHEDULE = [15.0] * DEFAULT_NUM_TIMEPOINTS  # simpler constant FA schedule
DEFAULT_LAC_FA_SCHEDULE = [15.0] * DEFAULT_NUM_TIMEPOINTS  # simpler constant FA schedule

def generate_variable_aif_params():
    return {'t0': 0, 'alpha': 3.0, 'beta': 1.0}
# def generate_variable_aif_params(rng=None):
#     if rng is None:
#         rng = np.random.RandomState()
#     t0 = rng.uniform(0, 5)  # AIF start time between 0 and 5 seconds
#     alpha = rng.uniform(1.0, 5.0)  # AIF shape parameter alpha
#     beta = rng.uniform(0.5, 2.0)   # AIF shape parameter beta
#     return {'t0': t0, 'alpha': alpha, 'beta': beta}

class TwoCompartmentHPDataGenerator:
    """
    Two-compartment HP 13C generator with VFA support.

    Flip angles for pyruvate and lactate can be either:
      - scalar (constant FA, legacy behavior), or
      - 1D arrays of length T (variable FA, one angle per TR)

    If schedules are arrays and time_points is None, time_points is inferred as:
        time_points = [0, TR, 2TR, ..., (T-1)TR]
    RF loss within the ODE is applied piecewise-constantly per TR:
        RF_loss(theta_k) = -ln(cos(theta_k)) / TR
    """

    def __init__(self,
                 vb_range=(VB_MIN, VB_MAX), kpl_range=(KPL_MIN, KPL_MAX), kve_range=(KVE_MIN, KVE_MAX),
                 r1p_range=(1/30, 1/30), r1l_range=(1/25, 1/25),
                 time_points=None,
                 flip_angle_pyr_deg=DEFAULT_PYR_FA_SCHEDULE,
                 flip_angle_lac_deg=DEFAULT_LAC_FA_SCHEDULE,
                 TR=DEFAULT_TR,
                 rng=None,
                 seed=None):
        self.vb_range = vb_range
        self.kpl_range = kpl_range
        self.kve_range = kve_range
        self.r1p_range = r1p_range
        self.r1l_range = r1l_range
        self.r1p = r1p_range[0]
        self.r1l = r1l_range[0]
        self.TR = float(TR)
        self.rng = rng if rng is not None else np.random.RandomState(seed)

        # Normalize time grid and schedules
        (self.time_points,
         self.theta_pyr_sched_deg,
         self.theta_lac_sched_deg) = self._prepare_time_and_schedule(
             time_points, flip_angle_pyr_deg, flip_angle_lac_deg, self.TR
        )

        # Back-compat scalar attributes (first value of schedule)
        self.flip_angle_pyr_deg = float(self.theta_pyr_sched_deg[0])
        self.flip_angle_lac_deg = float(self.theta_lac_sched_deg[0])
        self.flip_angle_pyr_rad = np.deg2rad(self.flip_angle_pyr_deg)
        self.flip_angle_lac_rad = np.deg2rad(self.flip_angle_lac_deg)

    # -------------------- Helpers --------------------

    @staticmethod
    def _to_1d_array(x):
        arr = np.atleast_1d(np.array(x, dtype=float))
        return arr

    def _prepare_time_and_schedule(self, time_points, th_pyr_deg, th_lac_deg, TR):
        th_p = self._to_1d_array(th_pyr_deg)
        th_l = self._to_1d_array(th_lac_deg)

        if th_p.shape != th_l.shape:
            raise ValueError("flip_angle_pyr_deg and flip_angle_lac_deg must have the same length.")

        T = th_p.size

        if time_points is None:
            # Infer time grid from schedule length and TR
            t = np.arange(T, dtype=float) * float(TR)
        else:
            t = np.array(time_points, dtype=float).ravel()
            if t.size != T:
                # If a scalar schedule was passed, broadcast it to match time_points
                if T == 1:
                    th_p = np.full_like(t, float(th_p.item()), dtype=float)
                    th_l = np.full_like(t, float(th_l.item()), dtype=float)
                else:
                    raise ValueError(f"Length mismatch: time_points={t.size} vs schedule={T}")

        return t, th_p, th_l

    @staticmethod
    def _rf_loss_per_TR(theta_deg, TR):
        """RF loss rate per second for a single flip angle (degrees)."""
        theta = np.deg2rad(theta_deg)
        return -np.log(np.clip(np.cos(theta), 1e-8, 1.0)) / float(TR)

    def _rf_losses_at_time(self, t):
        """
        Return (rf_loss_p, rf_loss_l) at continuous time t by selecting the
        piecewise-constant TR bin based on self.time_points.
        """
        t0 = float(self.time_points[0])
        if t <= t0:
            idx = 0
        else:
            idx = int(np.floor((t - t0) / self.TR))
        idx = int(np.clip(idx, 0, len(self.time_points) - 1))

        rfp = self._rf_loss_per_TR(self.theta_pyr_sched_deg[idx], self.TR)
        rfl = self._rf_loss_per_TR(self.theta_lac_sched_deg[idx], self.TR)

        return rfp, rfl

    # -------------------- Physiology --------------------

    @staticmethod
    def _aif(t, t0, alpha, beta):
        t_shifted = np.maximum(t - t0, 0.0)
        return alpha * t_shifted * np.exp(-beta * t_shifted)

    def _sample_params(self):
        return {
            'vb':  self.rng.uniform(*self.vb_range),
            'kpl': self.rng.uniform(*self.kpl_range),
            'kve': self.rng.uniform(*self.kve_range),
            'r1p': self.r1p,
            'r1l': self.r1l,
            **generate_variable_aif_params()
        }

    # -------------------- Model & Simulation --------------------

    def _solve_2c_model(self, params):
        """
        Two-compartment model with extravascular pyruvate (Pe) and lactate (Le),
        arterial/intravascular pyruvate Pv given by AIF, and time-varying RF loss.
        """
        def deriv(y, t):
            Pe, Le = y
            AIF = self._aif(t, params['t0'], params['alpha'], params['beta'])
            rf_p, rf_l = self._rf_losses_at_time(t)

            dPe_dt = AIF - (params['kpl'] + params['kve'] + params['r1p'] + rf_p) * Pe
            dLe_dt = params['kpl'] * Pe - (params['r1l'] + rf_l) * Le
            return [dPe_dt, dLe_dt]

        y0 = [0.0, 0.0]
        sol = odeint(deriv, y0, self.time_points)
        Pe, Le = sol[:, 0], sol[:, 1]
        Pv = self._aif(self.time_points, params['t0'], params['alpha'], params['beta'])

        # Total measured signals (simple mixture model)
        S_pyr = params['vb'] * Pv + (1.0 - params['vb']) * Pe
        S_lac = (1.0 - params['vb']) * Le

        return {'S_pyr': S_pyr, 'S_lac': S_lac, 'time': self.time_points}


    def generate_dataset(self, n_samples=1000, noise_std=0.05):
        X = []
        y = []
        for _ in range(n_samples):
            params = self._sample_params()
            result = self._solve_2c_model(params)
            S_pyr_noisy = result['S_pyr'] + self.rng.normal(0.0, noise_std, size=len(result['S_pyr']))
            S_lac_noisy = result['S_lac'] + self.rng.normal(0.0, noise_std, size=len(result['S_lac']))
            # shape (T, 2): [pyr, lac] per timepoint
            X.append(np.stack([S_pyr_noisy, S_lac_noisy], axis=1))
            y.append([params['kpl'], params['kve'], params['vb']])
        return np.array(X), np.array(y)





