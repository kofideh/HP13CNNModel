
import numpy as np

from hybrid_model_utils import VB_MIN, VB_MAX, KPL_MIN, KPL_MAX, KVE_MIN, KVE_MAX


DEFAULT_VFA_SCHEDULE = np.array([
    14.4775, 14.9632, 15.5014, 16.1021, 16.7787, 17.5484, 18.4349, 19.4712,
    20.7048, 22.2077, 24.0948, 26.5651, 30.0000, 35.2644, 45.0000, 89.9
], dtype=float)

class TwoCompartmentHPDataGeneratorMeasured:
    """
    Measured-pyruvate driver generator (stable exponential integrator).

    Signals:
      S_pyr(t)  = drawn measured pyruvate timecourse (driver)
      S_lac(t)  = (1 - vb) * L(t),  L' = kPL * Pe - R1Lapp(t) * L
      Pe'       = (kVE/vb)*S_pyr - [kVE*(1-vb)/vb + kPL + R1Papp(t)] * Pe

    RF loss per TR: RF = -ln(max(cos(theta), eps)) / TR, with theta capped < 90 deg.
    """
    def __init__(self,
                 vb_range=(VB_MIN, VB_MAX),
                 kpl_range=(KPL_MIN, KPL_MAX),
                 kve_range=(KVE_MIN, KVE_MAX),
                 r1p_range=(1/30, 1/30),
                 r1l_range=(1/25, 1/25),
                 time_points=None,
                 flip_angle_pyr_deg=DEFAULT_VFA_SCHEDULE,
                 flip_angle_lac_deg=DEFAULT_VFA_SCHEDULE,
                 TR=2.0,
                 seed=None):
        self.rng = np.random.default_rng(seed)
        self.vb_range = vb_range
        self.kpl_range = kpl_range
        self.kve_range = kve_range
        self.r1p_range = r1p_range
        self.r1l_range = r1l_range
        self.TR = float(TR)

        # prep time grid & FA schedules
        th_p = np.array(flip_angle_pyr_deg, dtype=float).ravel()
        th_l = np.array(flip_angle_lac_deg, dtype=float).ravel()
        if th_p.shape != th_l.shape:
            raise ValueError("flip_angle_pyr_deg and flip_angle_lac_deg must have same length")

        self.T = th_p.size
        if time_points is None:
            self.time_points = np.arange(self.T, dtype=float) * self.TR
        else:
            t = np.array(time_points, dtype=float).ravel()
            if t.size != self.T:
                raise ValueError("time_points length must match flip schedule length")
            self.time_points = t

        # cap to < 90 to avoid infinite RF losses
        self.theta_pyr_sched_deg = np.minimum(th_p, 89.9)
        self.theta_lac_sched_deg = np.minimum(th_l, 89.9)

    # ----------- helpers -----------
    @staticmethod
    def _rf_loss_per_TR(theta_deg, TR):
        theta_rad = np.deg2rad(theta_deg)
        cosv = np.cos(theta_rad)
        cosv = np.clip(cosv, 1e-6, 1.0)  # avoid cos ~ 0 at 90 deg
        return -np.log(cosv) / float(TR)

    @staticmethod
    def _exp_update_linear(dt, a, b, x_prev):
        """
        Solve dx/dt = a - b x, with a,b constants over [t, t+dt].
        Exact solution:
            x_next = x_prev * exp(-b dt) + (a/b) * (1 - exp(-b dt))  (if b>0)
        Handles small b via series.
        """
        b = float(b)
        if b <= 1e-9:
            # near zero-loss: Euler is fine
            return x_prev + dt * (a - b * x_prev)
        e = np.exp(-b * dt)
        return x_prev * e + (a / b) * (1.0 - e)

    def _piecewise_R1app(self, flips_deg, R1_base):
        # RF loss per TR as piecewise-constant
        rf = self._rf_loss_per_TR(flips_deg, self.TR)
        return R1_base + rf

    def _sample_params(self):
        def r(lo, hi): return self.rng.uniform(lo, hi)
        vb  = r(*self.vb_range)
        kpl = r(*self.kpl_range)
        kve = r(*self.kve_range)
        r1p = r(*self.r1p_range)
        r1l = r(*self.r1l_range)
        return dict(vb=vb, kpl=kpl, kve=kve, r1p=r1p, r1l=r1l)

    def _draw_Spyr(self):
        t = self.time_points
        # gamma-variate–like but bounded; choose parameters for realistic scale
        amp   = self.rng.uniform(0.5, 1.5)
        alpha = self.rng.uniform(2.0, 4.0)
        beta  = self.rng.uniform(0.6, 1.2)
        t0    = self.rng.uniform(0.0, 2.0)
        tt = np.maximum(t - t0, 0.0)
        Sp = amp * (tt ** alpha) * np.exp(-beta * tt)
        # normalize to a plausible peak (0.7–1.3), then rescale
        peak = Sp.max() + 1e-12
        scale = self.rng.uniform(0.7, 1.3) / peak
        return np.clip(Sp * scale, 0.0, None)

    def _simulate_measured_driver(self, params):
        t = self.time_points
        dt = np.diff(t, prepend=t[0])
        vb = float(np.clip(params["vb"], 0.01, 0.99))

        R1app_P = self._piecewise_R1app(self.theta_pyr_sched_deg, params["r1p"])
        R1app_L = self._piecewise_R1app(self.theta_lac_sched_deg, params["r1l"])

        Sp = self._draw_Spyr()
        Pe = np.zeros_like(Sp)
        L  = np.zeros_like(Sp)

        # constants per TR
        gain_const = params["kve"] / vb
        loss_const_base = params["kve"] * (1.0 - vb) / vb + params["kpl"]

        for k in range(1, t.size):
            # Pe step: a - b Pe with a = gain*Sp[k-1], b = loss_base + R1app_P[k-1]
            a_pe = gain_const * Sp[k-1]
            b_pe = loss_const_base + R1app_P[k-1]
            Pe[k] = self._exp_update_linear(dt[k], a_pe, b_pe, Pe[k-1])

            # L step: a - b L with a = kpl * Pe[k-1 or k?]; use Pe[k-1] (left-const)
            a_l = params["kpl"] * Pe[k-1]
            b_l = R1app_L[k-1]
            L[k] = self._exp_update_linear(dt[k], a_l, b_l, L[k-1])

        S_pyr = Sp
        S_lac = (1.0 - vb) * L
        return S_pyr, S_lac

    def generate_dataset(self, n_samples=1000, noise_std=0.02):
        X = []
        y = []
        for _ in range(int(n_samples)):
            p = self._sample_params()
            Sp, Sl = self._simulate_measured_driver(p)
            if noise_std and noise_std > 0:
                Sp = Sp + self.rng.normal(0.0, noise_std, size=Sp.shape)
                Sl = Sl + self.rng.normal(0.0, noise_std, size=Sl.shape)
            X.append(np.stack([Sp, Sl], axis=1))
            y.append([p["kpl"], p["kve"], p["vb"]])
        return np.array(X), np.array(y)
