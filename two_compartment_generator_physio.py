
import numpy as np
from scipy.stats import loguniform


class TwoCompartmentHPDataGeneratorPhysio:
    """
    Physiologic two-compartment HP 13C pyruvate/lactate generator.

    Vascular input:
        Pv(t) = gamma-variate-like vascular pyruvate input

    Dynamics:
        dPe/dt = kVE*Pv - (kVE + kPL + R1Papp)*Pe
        dL/dt  = kPL*Pe - R1Lapp*L

    Observed voxel signals:
        S_pyr(t) = vb*Pv(t) + (1-vb)*Pe(t)
        S_lac(t) = (1-vb)*L(t)
    """
    def __init__(self,
                vb_range=None, kpl_range=None, kve_range=None,
                r1p_range=(1/30, 1/30), r1l_range=(1/25, 1/25),
                time_points=None,
                flip_angle_pyr_deg=None,
                flip_angle_lac_deg=None,
                TR=None,
                rng=None,
                seed=None):
        self.rng = rng if rng is not None else np.random.RandomState(seed)
        self.vb_range = vb_range
        self.kpl_range = kpl_range
        self.kve_range = kve_range
        self.r1p_range = r1p_range
        self.r1l_range = r1l_range
        self.TR = float(TR)

        # prep time grid & FA schedules
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
        theta_deg = np.minimum(np.asarray(theta_deg, dtype=float), 89.9)
        theta_rad = np.deg2rad(theta_deg)
        cosv = np.cos(theta_rad)
        cosv = np.clip(cosv, 1e-8, 1.0)
        return -np.log(cosv) / float(TR)

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

    def _draw_vascular_input(self):
        """
        Draw an effective vascular pyruvate input Pv(t), represented by a
        gamma-variate-like bolus curve sampled at the acquisition time points.

        This is not the measured voxel pyruvate signal S_pyr(t). The measured
        pyruvate signal is generated later as:

            S_pyr(t) = vb * Pv(t) + (1 - vb) * Pe(t)
        """
        t = self.time_points
        # gamma-variate–like but bounded; choose parameters for realistic scale
        peak_amp = self.rng.uniform(0.7, 1.3)
        alpha = self.rng.uniform(2.0, 4.0)
        beta  = self.rng.uniform(0.6, 1.2)
        t0    = self.rng.uniform(0.0, 2.0)
        tau = np.maximum(t - t0, 0.0)
        Pv = (tau ** alpha) * np.exp(-beta * tau)
        peak = Pv.max() 
        if peak <= 1e-12:
            return np.zeros_like(t)
        
        Pv = peak_amp * Pv / peak
        return np.clip(Pv, 0.0, None)

    def _simulate_physio_model(self, params):
        t = self.time_points
        dt = np.diff(t, prepend=t[0])
        vb = float(np.clip(params["vb"], 0.01, 0.99))

        R1app_P = self._piecewise_R1app(self.theta_pyr_sched_deg, params["r1p"])
        R1app_L = self._piecewise_R1app(self.theta_lac_sched_deg, params["r1l"])

        Pv = self._draw_vascular_input()
        Pe = np.zeros_like(Pv)
        L  = np.zeros_like(Pv)

        for k in range(1, t.size):
            # Pe' = kve*Pv - (kve + kpl + R1Papp)*Pe
            a_pe = params["kve"] * Pv[k-1]
            b_pe = params["kve"] + params["kpl"] + R1app_P[k-1]
            Pe[k] = self._exp_update_linear(dt[k], a_pe, b_pe, Pe[k-1])

            # L' = kpl*Pe - R1Lapp*L
            a_l = params["kpl"] * Pe[k-1]
            b_l = R1app_L[k-1]
            L[k] = self._exp_update_linear(dt[k], a_l, b_l, L[k-1])

        S_pyr = vb * Pv + (1.0 - vb) * Pe
        S_lac = (1.0 - vb) * L
        return S_pyr, S_lac, Pv, Pe, L


    def generate_dataset(self, n_samples=1000, snr_range=None, noise_std=None):
        X = []
        y = []
        met_pools = []
        noise_info = []

        for _ in range(int(n_samples)):
            p = self._sample_params()
            Sp, Sl, Pv, Pe, L = self._simulate_physio_model(p)
            met_pools.append(np.stack([Pv, Pe, L], axis=1))

            if snr_range is not None:
                # Draw one target SNR per synthetic example.
                # Example: snr_range=(20, 100)
                target_snr = np.exp(
                    self.rng.uniform(
                        np.log(snr_range[0]),
                        np.log(snr_range[1])
                    )
                )

                # Define noise relative to pyruvate peak.
                # This gives approximately peak-pyruvate SNR = target_snr.
                peak_ref = max(np.max(np.abs(Sp)), 1e-8)
                sigma = peak_ref / target_snr

                Sp = Sp + self.rng.normal(0.0, sigma, size=Sp.shape)
                Sl = Sl + self.rng.normal(0.0, sigma, size=Sl.shape)

                noise_info.append([target_snr, sigma])

            elif noise_std is not None and noise_std > 0:
                Sp = Sp + self.rng.normal(0.0, noise_std, size=Sp.shape)
                Sl = Sl + self.rng.normal(0.0, noise_std, size=Sl.shape)

                peak_ref = max(np.max(np.abs(Sp)), 1e-8)
                noise_info.append([peak_ref / noise_std, noise_std])

            else:
                noise_info.append([np.inf, 0.0])

            X.append(np.stack([Sp, Sl], axis=1))
            y.append([p["kpl"], p["kve"], p["vb"]])

        return np.array(X), np.array(y), np.array(met_pools), np.array(noise_info)
