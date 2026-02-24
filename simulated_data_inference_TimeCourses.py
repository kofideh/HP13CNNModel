import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import pandas as pd
from hybrid_model_utils import (
    HybridMultiHead, plot_prediction_scatter2,
    VB_MIN, VB_MAX, KPL_MIN, KPL_MAX, KVE_MIN, KVE_MAX
) 
from two_compartment_generator import (
    TwoCompartmentHPDataGenerator, DEFAULT_TR, DEFAULT_NUM_TIMEPOINTS, PYR_FA_SCHEDULE, LAC_FA_SCHEDULE,
)   
n_samples = 50


# 1. Generate data
time_points=np.arange(0, DEFAULT_NUM_TIMEPOINTS * DEFAULT_TR, DEFAULT_TR) # 16 time points from 0 to 30s with TR=2s
n_timepoints = len(time_points)
noise = 0.05  # Example noise level
generator = TwoCompartmentHPDataGenerator(time_points=time_points,
                                              vb_range=(VB_MIN, VB_MAX),
                                                kpl_range=(KPL_MIN, KPL_MAX),
                                                kve_range=(KVE_MIN, KVE_MAX))
X, y = generator.generate_dataset(n_samples=n_samples, noise_std=noise)

# 2. Split channels
X_raw = X.copy()  # raw data for vb
X_norm = X 
n_output_params = 3

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
run_dir = os.path.join(output_dir, "wts_2C_GammaAIF_noisestd0.05")
weights_path = os.path.join(run_dir, "trained_hybrid_positive.pth")
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join(run_dir, f"simulatedData_{timestamp}")
os.makedirs(run_dir, exist_ok=True)




# Plot comparison of X_raw vs X_norm
import matplotlib.pyplot as plt

# Select a few random samples to plot
n_samples_to_plot = 5
sample_indices = np.random.choice(X.shape[0], n_samples_to_plot, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Comparison of X_raw vs X_norm', fontsize=16)

for i, sample_idx in enumerate(sample_indices):
    if i >= 5:  # Only plot first 5 samples
        break
    
    row = i // 3
    col = i % 3
    
    # Plot pyruvate (channel 0)
    axes[row, col].plot(time_points, X_raw[sample_idx, :, 0], 'b-', label='Pyruvate (Raw)', linewidth=2)
    axes[row, col].plot(time_points, X_raw[sample_idx, :, 1], 'r-', label='Lactate (Raw)', linewidth=2)
    
    # Plot normalized on secondary y-axis
    ax2 = axes[row, col].twinx()
    ax2.plot(time_points, X_norm[sample_idx, :, 0], 'b--', label='Pyruvate (Norm)', alpha=0.7)
    ax2.plot(time_points, X_norm[sample_idx, :, 1], 'r--', label='Lactate (Norm)', alpha=0.7)
    
    axes[row, col].set_xlabel('Time (s)')
    axes[row, col].set_ylabel('Raw Signal', color='black')
    ax2.set_ylabel('Normalized Signal', color='gray')
    axes[row, col].set_title(f'Sample {sample_idx + 1}')
    axes[row, col].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[row, col].grid(True, alpha=0.3)

# Hide the last subplot if we have fewer than 6 samples
if n_samples_to_plot < 6:
    axes[1, 2].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "X_raw_vs_X_norm_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# Create histogram comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of all values in X_raw
axes[0].hist(X_raw.flatten(), bins=50, alpha=0.7, color='blue', label='X_raw')
axes[0].set_xlabel('Signal Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of X_raw Values')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Histogram of all values in X_norm
axes[1].hist(X_norm.flatten(), bins=50, alpha=0.7, color='red', label='X_norm')
axes[1].set_xlabel('Signal Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of X_norm Values')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "X_raw_vs_X_norm_histograms.png"), dpi=150, bbox_inches='tight')
plt.close()

# 3. Flatten inputs
Xr_test = X_raw.reshape(X.shape[0], -1)
Xn_test = X_norm.reshape(X.shape[0], -1)
Xr_test_tensor = torch.tensor(Xr_test, dtype=torch.float32)
Xn_test_tensor = torch.tensor(Xn_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y, dtype=torch.float32)

# 1. Load trained model
input_dim_raw = DEFAULT_NUM_TIMEPOINTS*2  # e.g., timepoints x 2 channels (pyr/lac) flattened
input_dim_norm = DEFAULT_NUM_TIMEPOINTS*2
model = HybridMultiHead(input_dim_raw=input_dim_raw, input_dim_norm=input_dim_norm)
model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
model.eval()


noise_levels = [0.05]


# cov shape: (n_noise_levels, model_type(0=NN,1=Traditional), param_index(kPL=0,kVE=1,vB=2))
cov = np.zeros((len(noise_levels), 2, 3))
snrs = np.zeros((len(noise_levels), 2))
# Arrays to hold mean predicted parameters per noise level (NN and Traditional)
mean_pred_nn = np.full((len(noise_levels), 3), np.nan)  # kPL,kVE,vB
mean_pred_tr = np.full((len(noise_levels), 3), np.nan)
# Arrays to hold coefficient of determination (R²) per noise level
r2_nn = np.full((len(noise_levels), 3), np.nan)
r2_tr = np.full((len(noise_levels), 3), np.nan)
# Bootstrap std estimates for R² (for shading)
r2_nn_std = np.full((len(noise_levels), 3), np.nan)
r2_tr_std = np.full((len(noise_levels), 3), np.nan)

for noise_idx, noise in enumerate(noise_levels):
    r2_kpl = 0
    r2_kve = 0
    r2_vb  = 0


    X, y = generator.generate_dataset(n_samples=n_samples, noise_std=noise)
    X_raw = X.copy()  # raw data for vb
    X_norm = X 
    # 3. Flatten inputs
    Xr_test = X_raw.reshape(X.shape[0], -1)
    Xn_test = X_norm.reshape(X.shape[0], -1)
    Xr_test_tensor = torch.tensor(Xr_test, dtype=torch.float32)
    Xn_test_tensor = torch.tensor(Xn_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y, dtype=torch.float32)   

    pyr_signals = X[:, :, 0]
    lac_signals = X[:, :, 1]

    # Compute SNR as mean amplitude divided by noise std
    snr_pyr = np.max(np.abs(pyr_signals)) / noise
    snr_lac = np.max(np.abs(lac_signals)) / noise
    snrs[noise_idx, 0] = snr_pyr
    snrs[noise_idx, 1] = snr_lac


    with torch.no_grad():
        y_pred = model(Xn_test_tensor, Xr_test_tensor).numpy()
        y_true = y_test_tensor.numpy()

        r2_kpl = 1 - np.sum((y_pred[:, 0] - y_true[:, 0])**2) / np.sum((y_true[:, 0] - np.mean(y_true[:, 0]))**2)
        r2_kve = 1 - np.sum((y_pred[:, 1] - y_true[:, 1])**2) / np.sum((y_true[:, 1] - np.mean(y_true[:, 1]))**2)
        r2_vb  = 1 - np.sum((y_pred[:, 2] - y_true[:, 2])**2) / np.sum((y_true[:, 2] - np.mean(y_true[:, 2]))**2)

        # Plot true vs. predicted for neural network
        import matplotlib.pyplot as plt
        def plot_true_vs_pred(y_true, y_pred, title_suffix):
            import matplotlib.pyplot as plt
            import numpy as np

            labels = ['kPL', 'kVE', 'vB']
            plt.figure(figsize=(15, 4))
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
                plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                        [y_true[:, i].min(), y_true[:, i].max()], 'r--')

                # Compute R²
                ss_res = np.sum((y_pred[:, i] - y_true[:, i]) ** 2)
                ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
                r2 = 1 - ss_res / ss_tot

                plt.xlabel("True")
                plt.ylabel("Predicted")
                plt.title(f"{labels[i]}: R²={r2:.3f}\n{title_suffix}")

            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f"true_vs_pred_{title_suffix}.png"))
            plt.close()
            import pandas as pd
            df = pd.DataFrame({
                'kPL_true': y_true[:, 0],
                'kVE_true': y_true[:, 1],
                'vB_true': y_true[:, 2],
                'kPL_pred': y_pred[:, 0],
                'kVE_pred': y_pred[:, 1],
                'vB_pred': y_pred[:, 2],
            })
            df.to_csv(os.path.join(run_dir, f"predictions_vs_truth_{title_suffix}.csv"), index=False)

        plot_true_vs_pred(y_true, y_pred, f"{snr_pyr}__{snr_lac}_Neural Network")
        
        
    # # === Plotting ===
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = os.path.join(run_dir, f'NeuralNetwork_SNR_PL_{int(snr_pyr)}_{int(snr_lac)}')
    plot_prediction_scatter2(y_true, y_pred, saveName=fname)

    # Compute CoV (coefficient of variation) robustly for neural-network predictions
    # per-parameter (kPL=0,kVE=1,vB=2). Use nan-aware stats and guard against small means.
    y_pred_mean = np.nanmean(y_pred, axis=0)
    y_pred_std = np.nanstd(y_pred, axis=0)
    eps = 1e-12
    for p_idx in range(3):
        mean_p = y_pred_mean[p_idx]
        std_p = y_pred_std[p_idx]
        if np.isfinite(mean_p) and np.abs(mean_p) > eps:
            cov[noise_idx, 0, p_idx] = std_p / np.abs(mean_p)
        else:
            cov[noise_idx, 0, p_idx] = np.nan
    # Store mean NN predictions (per-parameter) for this noise level
    if y_pred.shape[1] >= 3:
        mean_pred_nn[noise_idx, :] = np.nanmean(y_pred, axis=0)[:3]
    else:
        tmp = np.nanmean(y_pred, axis=0)
        mean_pred_nn[noise_idx, :tmp.shape[0]] = tmp
    # Compute per-parameter R^2 (NN)
    for p_idx in range(3):
        num = np.nansum((y_pred[:, p_idx] - y_true[:, p_idx])**2)
        den = np.nansum((y_true[:, p_idx] - np.nanmean(y_true[:, p_idx]))**2)
        if den > 0:
            r2_nn[noise_idx, p_idx] = 1 - num/den
        else:
            r2_nn[noise_idx, p_idx] = np.nan

    # Bootstrap estimate of R^2 std (NN)
    try:
        B = 200
        r2b = np.full((B, 3), np.nan)
        n_samples_local = y_true.shape[0]
        if n_samples_local > 1:
            for b in range(B):
                idxs = np.random.randint(0, n_samples_local, size=n_samples_local)
                y_t_b = y_true[idxs]
                y_p_b = y_pred[idxs]
                for p_idx in range(3):
                    den_b = np.nansum((y_t_b[:, p_idx] - np.nanmean(y_t_b[:, p_idx]))**2)
                    if den_b > 0:
                        num_b = np.nansum((y_p_b[:, p_idx] - y_t_b[:, p_idx])**2)
                        r2b[b, p_idx] = 1 - num_b/den_b
            r2_nn_std[noise_idx, :] = np.nanstd(r2b, axis=0)
    except Exception:
        pass

    # # Traditional fit comparison
    traditional_preds = []


    
    from fit_two_compartment import fit_traditional_2c_model


    X_test = X

    print("Using traditional fittings...")
    import time
    start_time = time.time()
    for j in range(len(X_test)):
        pyr = X_test[j, :, 0]
        lac = X_test[j, :, 1]
        try:
            params, _, success = fit_traditional_2c_model(
                time_points, pyr, lac, estimate_r1=False, 
                flip_angle_pyr_deg=PYR_FA_SCHEDULE, flip_angle_lac_deg=LAC_FA_SCHEDULE, TR=DEFAULT_TR,
                use_3state=(n_output_params==4))
            if n_output_params == 4:
                traditional_preds.append(params[:4])  # [kpl, kve, vb, Ktrans]
            else:
                traditional_preds.append(params[:3])  # [kpl, kve, vb]
            # res = fit_measured_driver_curvefit(
            #     time=time_points,
            #     S_pyr=pyr,
            #     S_lac=lac,
            #     theta_pyr_deg=PYR_FA_SCHEDULE,   # e.g. your DEFAULT_VFA_SCHEDULE or a constant
            #     theta_lac_deg=LAC_FA_SCHEDULE,
            #     TR=DEFAULT_TR,
            #     R1p=1/30.0,
            #     R1l=1/25.0,
            # )
            # traditional_preds.append([res["kpl"], res["kve"], res["vb"]])
            # res = fit_two_stage_nlls(
            #     time=time_points,                      # (T,)
            #     S_pyr=pyr,                 # measured pyruvate signal
            #     S_lac=lac,                 # measured lactate signal
            #     theta_pyr_deg=DEFAULT_VFA_SCHEDULE,       # scalar or (T,)
            #     theta_lac_deg=DEFAULT_VFA_SCHEDULE,       # scalar or (T,)
            #     TR=2.0,
            #     R1p=1/30.0,
            #     R1l=1/25.0,
            #     fix_vb=True,                 # fix vB to stage-1 estimate
            #     fix_kve=False                # optionally also fix kVE
            # )
            # traditional_preds.append([res["kpl"], res["kve"], res["vb"]])
        except:
            traditional_preds.append([np.nan] * n_output_params)
    tradfit_time = time.time() - start_time
    print(f"Traditional model training time: {tradfit_time:.2f} seconds")
    y_pred_fit = np.array(traditional_preds)
    fname = os.path.join(run_dir, f'TraditionalFitOfGeneratorData_SNR_PL_{int(snr_pyr)}_{int(snr_lac)}')
    plot_prediction_scatter2(y_true, y_pred_fit, saveName=fname)
    # CoV for traditional (NLLS) fits: per-parameter using nan-aware statistics
    y_pred_mean = np.nanmean(y_pred_fit, axis=0)
    y_pred_std = np.nanstd(y_pred_fit, axis=0)
    eps = 1e-12
    # If y_pred_fit has fewer than 3 columns (rare), pad with NaN
    if y_pred_mean.shape[0] < 3:
        y_pred_mean = np.pad(y_pred_mean, (0, 3 - y_pred_mean.shape[0]), constant_values=np.nan)
        y_pred_std = np.pad(y_pred_std, (0, 3 - y_pred_std.shape[0]), constant_values=np.nan)
    for p_idx in range(3):
        mean_p = y_pred_mean[p_idx]
        std_p = y_pred_std[p_idx]
        if np.isfinite(mean_p) and np.abs(mean_p) > eps:
            cov[noise_idx, 1, p_idx] = std_p / np.abs(mean_p)
        else:
            cov[noise_idx, 1, p_idx] = np.nan
    # Store mean Traditional predictions (per-parameter) for this noise level
    if y_pred_fit.ndim == 2 and y_pred_fit.shape[1] >= 3:
        mean_pred_tr[noise_idx, :] = np.nanmean(y_pred_fit, axis=0)[:3]
    else:
        # if fits returned fewer params, pad with NaN
        tmp = np.nanmean(y_pred_fit, axis=0)
        for ii in range(min(len(tmp), 3)):
            mean_pred_tr[noise_idx, ii] = tmp[ii]
    # Compute per-parameter R^2 (Traditional)
    if y_pred_fit.ndim == 2:
        for p_idx in range(min(3, y_pred_fit.shape[1])):
            num = np.nansum((y_pred_fit[:, p_idx] - y_true[:, p_idx])**2)
            den = np.nansum((y_true[:, p_idx] - np.nanmean(y_true[:, p_idx]))**2)
            if den > 0:
                r2_tr[noise_idx, p_idx] = 1 - num/den
            else:
                r2_tr[noise_idx, p_idx] = np.nan

    # Bootstrap estimate of R^2 std (Traditional)
    try:
        if y_pred_fit.ndim == 2 and y_pred_fit.shape[0] > 1:
            B = 200
            r2b = np.full((B, 3), np.nan)
            n_samples_local = y_true.shape[0]
            for b in range(B):
                idxs = np.random.randint(0, n_samples_local, size=n_samples_local)
                y_t_b = y_true[idxs]
                y_pf_b = y_pred_fit[idxs]
                for p_idx in range(min(3, y_pred_fit.shape[1])):
                    den_b = np.nansum((y_t_b[:, p_idx] - np.nanmean(y_t_b[:, p_idx]))**2)
                    if den_b > 0 and y_pf_b.shape[0] > 0:
                        num_b = np.nansum((y_pf_b[:, p_idx] - y_t_b[:, p_idx])**2)
                        r2b[b, p_idx] = 1 - num_b/den_b
            r2_tr_std[noise_idx, :] = np.nanstd(r2b, axis=0)
    except Exception:
        pass

    # Plot example time courses with fits (for reviewer)
    # Select a few representative samples to show fits
    if noise_idx in [0, len(noise_levels)//2, len(noise_levels)-1]:  # Low, mid, high SNR
        n_examples = min(3, len(X_test))
        
        # Select examples with diverse kPL values (low, medium, high)
        kpl_true_values = y_true[:, 0]
        sorted_indices = np.argsort(kpl_true_values)
        example_indices = [
            sorted_indices[0],  # Lowest kPL
            sorted_indices[len(sorted_indices)//2],  # Medium kPL
            sorted_indices[-1]  # Highest kPL
        ]
        example_indices = example_indices[:n_examples]
        
        # Determine consistent y-axis ranges across all examples
        pyr_max = np.max(X_test[:, :, 0]) * 1.1
        lac_max = np.max(X_test[:, :, 1]) * 1.1
        
        fig, axes = plt.subplots(n_examples, 2, figsize=(14, 4*n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)
        
        # Helper function to reconstruct fitted curves
        def forward_model_2c(params, time_points, generator):
            """Generate synthetic signals from fitted parameters"""
            kpl, kve, vb = params[0], params[1], params[2]
            r1p, r1l = 1/30, 1/25
            t0, alpha, beta = 0, 3.0, 1.0
            
            def _aif(t, t0, alpha, beta):
                t_shifted = np.maximum(t - t0, 0.0)
                return alpha * t_shifted * np.exp(-beta * t_shifted)
            
            def deriv(y, t):
                Pe, Le = y
                AIF = _aif(t, t0, alpha, beta)
                rf_p, rf_l = generator._rf_losses_at_time(t)
                dPe_dt = AIF - (kpl + kve + r1p + rf_p) * Pe
                dLe_dt = kpl * Pe - (r1l + rf_l) * Le
                return [dPe_dt, dLe_dt]
            
            y0 = [0.0, 0.0]
            from scipy.integrate import odeint
            sol = odeint(deriv, y0, time_points)
            Pe, Le = sol[:, 0], sol[:, 1]
            Pv = _aif(time_points, t0, alpha, beta)
            S_pyr = vb * Pv + (1.0 - vb) * Pe
            S_lac = (1.0 - vb) * Le
            return S_pyr, S_lac
        
        for plot_idx, sample_idx in enumerate(example_indices):
            # Get observed data
            pyr_obs = X_test[sample_idx, :, 0]
            lac_obs = X_test[sample_idx, :, 1]
            
            # Get NN predictions
            nn_params = y_pred[sample_idx]
            kpl_nn, kve_nn, vb_nn = nn_params[0], nn_params[1], nn_params[2]
            
            # Get Traditional predictions
            tr_params = y_pred_fit[sample_idx]
            kpl_tr, kve_tr, vb_tr = tr_params[0], tr_params[1], tr_params[2]
            
            # True parameters
            true_params = y_true[sample_idx]
            kpl_true, kve_true, vb_true = true_params[0], true_params[1], true_params[2]
            
            # Generate fitted curves
            pyr_nn, lac_nn = forward_model_2c(nn_params, time_points, generator)
            pyr_tr, lac_tr = forward_model_2c(tr_params, time_points, generator)
            
            # Plot pyruvate
            ax = axes[plot_idx, 0]
            ax.plot(time_points, pyr_obs, 'ko', label='Observed', markersize=7, alpha=0.8, zorder=3)
            ax.plot(time_points, pyr_nn, 'b-', label='NN fit', linewidth=2.5, zorder=2)
            ax.plot(time_points, pyr_tr, 'r--', label='NLLS fit', linewidth=2.5, zorder=1)
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Signal (normalized intensity)', fontsize=12)
            ax.set_title(f'Pyruvate (SNR={snr_pyr:.1f})\n' + 
                        f'True: $k_{{PL}}$={kpl_true:.3f}, $k_{{VE}}$={kve_true:.3f}, $v_B$={vb_true:.3f}',
                        fontsize=11)
            ax.set_ylim(0, pyr_max)
            ax.grid(True, alpha=0.3)
            
            # Add text box with fitted parameters (outside the plot area via legend)
            param_text = (f'NN:    $k_{{PL}}$={kpl_nn:.3f}, $k_{{VE}}$={kve_nn:.3f}, $v_B$={vb_nn:.3f}\n'
                         f'NLLS: $k_{{PL}}$={kpl_tr:.3f}, $k_{{VE}}$={kve_tr:.3f}, $v_B$={vb_tr:.3f}')
            ax.text(0.98, 0.97, param_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Simple legend for line styles only
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
            
            # Plot lactate
            ax = axes[plot_idx, 1]
            ax.plot(time_points, lac_obs, 'ko', label='Observed', markersize=7, alpha=0.8, zorder=3)
            ax.plot(time_points, lac_nn, 'b-', label='NN fit', linewidth=2.5, zorder=2)
            ax.plot(time_points, lac_tr, 'r--', label='NLLS fit', linewidth=2.5, zorder=1)
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Signal (normalized intensity)', fontsize=12)
            ax.set_title(f'Lactate (SNR={snr_lac:.1f})\n' + 
                        f'True: $k_{{PL}}$={kpl_true:.3f}, $k_{{VE}}$={kve_true:.3f}, $v_B$={vb_true:.3f}',
                        fontsize=11)
            ax.set_ylim(0, lac_max)
            ax.grid(True, alpha=0.3)
            
            # Add text box with fitted parameters
            ax.text(0.98, 0.97, param_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Simple legend for line styles only
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f'Example_Timecourses_SNR_PL_{int(snr_pyr)}_{int(snr_lac)}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        plot_true_vs_pred(y_true, y_pred_fit, f"{snr_pyr}__{snr_lac}_TraditionalFit")


    
## Save SNRs and per-parameter CoV to Excel file
# Expand cov into separate 1-D columns so pandas DataFrame is happy
data_dict = {
    'Noise_Level': noise_levels,
    'SNR_Pyruvate': snrs[:, 0],
    'SNR_Lactate': snrs[:, 1],
    # Neural network CoV per parameter
    'COV_NN_kPL': cov[:, 0, 0],
    'COV_NN_kVE': cov[:, 0, 1],
    'COV_NN_vB':  cov[:, 0, 2],
    # Traditional fit CoV per parameter
    'COV_TR_kPL': cov[:, 1, 0],
    'COV_TR_kVE': cov[:, 1, 1],
    'COV_TR_vB':  cov[:, 1, 2],
}

df = pd.DataFrame(data_dict)
excel_filename = os.path.join(run_dir, f'SNR_COV_Analysis_{timestamp}.xlsx')
df.to_excel(excel_filename, index=False)
print(f"Data saved to {excel_filename}")

# Create R² summary table with uncertainties for manuscript
r2_summary_dict = {
    'Noise_Level': noise_levels,
    'SNR_Pyruvate': snrs[:, 0],
    'SNR_Lactate': snrs[:, 1],
    # NN R² with uncertainties
    'R2_NN_kPL': r2_nn[:, 0],
    'R2_NN_kPL_std': r2_nn_std[:, 0],
    'R2_NN_kVE': r2_nn[:, 1],
    'R2_NN_kVE_std': r2_nn_std[:, 1],
    'R2_NN_vB': r2_nn[:, 2],
    'R2_NN_vB_std': r2_nn_std[:, 2],
    # Traditional R² with uncertainties
    'R2_TR_kPL': r2_tr[:, 0],
    'R2_TR_kPL_std': r2_tr_std[:, 0],
    'R2_TR_kVE': r2_tr[:, 1],
    'R2_TR_kVE_std': r2_tr_std[:, 1],
    'R2_TR_vB': r2_tr[:, 2],
    'R2_TR_vB_std': r2_tr_std[:, 2],
}

df_r2 = pd.DataFrame(r2_summary_dict)
r2_excel_filename = os.path.join(run_dir, f'R2_Summary_with_Uncertainties_{timestamp}.xlsx')
df_r2.to_excel(r2_excel_filename, index=False)
print(f"R² summary saved to {r2_excel_filename}")

# Create formatted table for manuscript (text format)
print("\n" + "="*80)
print("R² VALUES WITH UNCERTAINTIES (for manuscript)")
print("="*80)
for noise_idx, noise in enumerate(noise_levels):
    print(f"\nNoise Level: {noise:.3f} | SNR (Pyr/Lac): {snrs[noise_idx, 0]:.1f}/{snrs[noise_idx, 1]:.1f}")
    print("-" * 80)
    print(f"{'Parameter':<10} {'NN R²':<25} {'NLLS R²':<25}")
    print("-" * 80)
    for p_idx, pname in enumerate(['kPL', 'kVE', 'vB']):
        nn_r2 = r2_nn[noise_idx, p_idx]
        nn_std = r2_nn_std[noise_idx, p_idx]
        tr_r2 = r2_tr[noise_idx, p_idx]
        tr_std = r2_tr_std[noise_idx, p_idx]
        
        nn_str = f"{nn_r2:.3f} ± {nn_std:.3f}" if np.isfinite(nn_std) else f"{nn_r2:.3f} ± N/A"
        tr_str = f"{tr_r2:.3f} ± {tr_std:.3f}" if np.isfinite(tr_std) else f"{tr_r2:.3f} ± N/A"
        
        print(f"{pname:<10} {nn_str:<25} {tr_str:<25}")

# Save formatted text table to file
with open(os.path.join(run_dir, f'R2_Manuscript_Table_{timestamp}.txt'), 'w') as f:
    f.write("R² VALUES WITH UNCERTAINTIES (Bootstrap estimate with B=200 iterations)\n")
    f.write("="*80 + "\n\n")
    for noise_idx, noise in enumerate(noise_levels):
        f.write(f"Noise Level: {noise:.3f} | SNR (Pyr/Lac): {snrs[noise_idx, 0]:.1f}/{snrs[noise_idx, 1]:.1f}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Parameter':<10} {'NN R²':<25} {'NLLS R²':<25}\n")
        f.write("-" * 80 + "\n")
        for p_idx, pname in enumerate(['kPL', 'kVE', 'vB']):
            nn_r2 = r2_nn[noise_idx, p_idx]
            nn_std = r2_nn_std[noise_idx, p_idx]
            tr_r2 = r2_tr[noise_idx, p_idx]
            tr_std = r2_tr_std[noise_idx, p_idx]
            
            nn_str = f"{nn_r2:.3f} ± {nn_std:.3f}" if np.isfinite(nn_std) else f"{nn_r2:.3f} ± N/A"
            tr_str = f"{tr_r2:.3f} ± {tr_std:.3f}" if np.isfinite(tr_std) else f"{tr_r2:.3f} ± N/A"
            
            f.write(f"{pname:<10} {nn_str:<25} {tr_str:<25}\n")
        f.write("\n")

print(f"\nFormatted manuscript table saved to: R2_Manuscript_Table_{timestamp}.txt")
print("="*80)

# Generate LaTeX table for manuscript
latex_table_file = os.path.join(run_dir, f'R2_LaTeX_Table_{timestamp}.tex')
with open(latex_table_file, 'w') as f:
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Coefficient of Determination ($R^2$) with Bootstrap Uncertainties for Neural Network and NLLS Methods}\n")
    f.write("\\label{tab:r2_uncertainties}\n")
    f.write("\\begin{tabular}{cccccccc}\n")
    f.write("\\hline\n")
    f.write("\\multirow{2}{*}{SNR (Pyr/Lac)} & \\multicolumn{3}{c}{Neural Network $R^2$} & & \\multicolumn{3}{c}{NLLS $R^2$} \\\\\n")
    f.write("\\cline{2-4} \\cline{6-8}\n")
    f.write(" & $k_{PL}$ & $k_{VE}$ & $v_B$ & & $k_{PL}$ & $k_{VE}$ & $v_B$ \\\\\n")
    f.write("\\hline\n")
    
    for noise_idx, noise in enumerate(noise_levels):
        snr_pyr = snrs[noise_idx, 0]
        snr_lac = snrs[noise_idx, 1]
        
        # Format R² values with uncertainties
        nn_kpl = f"{r2_nn[noise_idx, 0]:.3f} $\\pm$ {r2_nn_std[noise_idx, 0]:.3f}" if np.isfinite(r2_nn_std[noise_idx, 0]) else f"{r2_nn[noise_idx, 0]:.3f}"
        nn_kve = f"{r2_nn[noise_idx, 1]:.3f} $\\pm$ {r2_nn_std[noise_idx, 1]:.3f}" if np.isfinite(r2_nn_std[noise_idx, 1]) else f"{r2_nn[noise_idx, 1]:.3f}"
        nn_vb = f"{r2_nn[noise_idx, 2]:.3f} $\\pm$ {r2_nn_std[noise_idx, 2]:.3f}" if np.isfinite(r2_nn_std[noise_idx, 2]) else f"{r2_nn[noise_idx, 2]:.3f}"
        
        tr_kpl = f"{r2_tr[noise_idx, 0]:.3f} $\\pm$ {r2_tr_std[noise_idx, 0]:.3f}" if np.isfinite(r2_tr_std[noise_idx, 0]) else f"{r2_tr[noise_idx, 0]:.3f}"
        tr_kve = f"{r2_tr[noise_idx, 1]:.3f} $\\pm$ {r2_tr_std[noise_idx, 1]:.3f}" if np.isfinite(r2_tr_std[noise_idx, 1]) else f"{r2_tr[noise_idx, 1]:.3f}"
        tr_vb = f"{r2_tr[noise_idx, 2]:.3f} $\\pm$ {r2_tr_std[noise_idx, 2]:.3f}" if np.isfinite(r2_tr_std[noise_idx, 2]) else f"{r2_tr[noise_idx, 2]:.3f}"
        
        f.write(f"{snr_pyr:.0f}/{snr_lac:.0f} & {nn_kpl} & {nn_kve} & {nn_vb} & & {tr_kpl} & {tr_kve} & {tr_vb} \\\\\n")
    
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\begin{tablenotes}\n")
    f.write("\\small\n")
    f.write("\\item Uncertainties represent one standard deviation estimated via bootstrap resampling with replacement (B=200 iterations, n=50 samples per iteration).\n")
    f.write("\\item SNR: Signal-to-noise ratio for pyruvate (Pyr) and lactate (Lac) metabolites.\n")
    f.write("\\item NLLS: Non-linear least squares fitting method.\n")
    f.write("\\end{tablenotes}\n")
    f.write("\\end{table}\n")

print(f"LaTeX table saved to: {latex_table_file}")

# plot coefficient of variation vs SNR (kPL only)
import matplotlib.pyplot as plt

plt.figure()
ax = plt.gca()
# kPL CoV is parameter index 0
plt.plot(snrs[:, 0], cov[:, 0, 0], label='NN CoV (kPL)', marker='o', color='blue')
plt.plot(snrs[:, 0], cov[:, 1, 0], label='NLLS CoV (kPL)', marker='x', color='orange')
plt.xlabel('SNR')
plt.ylabel('Coefficient of Variation (kPL)')
plt.title('Coefficient of Variation (kPL) vs SNR')
plt.legend()

# Enable minor ticks on y-axis
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))  # Adjust spacing as needed
ax.tick_params(axis='y', which='minor', length=3, width=0.5)
ax.grid(True, which='minor', alpha=0.3, linestyle=':')
ax.grid(True, which='major', alpha=0.7)

plt.savefig(os.path.join(run_dir, 'Cov_vs_SNR_kPL.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Plot SNR vs R² (true vs predicted) for each parameter with error bars ---
params = ['kPL', 'kVE', 'vB']
for p_idx, pname in enumerate(params):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    x = snrs[:, 0]
    y_nn = r2_nn[:, p_idx]
    y_tr = r2_tr[:, p_idx]
    yerr_nn = r2_nn_std[:, p_idx]
    yerr_tr = r2_tr_std[:, p_idx]

    # Plot with error bars
    plt.errorbar(x, y_nn, yerr=yerr_nn, label=f'NN R² {pname}', marker='o', 
                 color='blue', capsize=5, capthick=2, linewidth=2, markersize=8)
    plt.errorbar(x, y_tr, yerr=yerr_tr, label=f'NLLS R² {pname}', marker='x', 
                 color='orange', capsize=5, capthick=2, linewidth=2, markersize=8)
    plt.xlabel('Pyruvate SNR', fontsize=12)
    plt.ylabel('Coefficient of Determination (R²)', fontsize=12)
    plt.title(f'Pyruvate SNR vs R² ({pname})', fontsize=13)
    plt.legend(fontsize=11)
    ax.set_ylim(-0.5, 1.05)
    ax.grid(True, which='major', alpha=0.7)
    ax.grid(True, which='minor', alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f'SNR_vs_R2_{pname}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# --- Combined plot: SNR vs R² for all parameters (NN and Traditional) with error bars ---
plt.figure(figsize=(10, 7))
ax = plt.gca()
colors = {'kPL': 'C0', 'kVE': 'C1', 'vB': 'C2'}
markers = {'NN': 'o', 'NLLS': 'x'}
for p_idx, pname in enumerate(params):
    x = snrs[:, 0]
    # Plot with error bars
    plt.errorbar(x, r2_nn[:, p_idx], yerr=r2_nn_std[:, p_idx], 
                 label=f'NN R² {pname}', marker=markers['NN'], color=colors[pname], 
                 linestyle='-', capsize=4, capthick=1.5, linewidth=2, markersize=7)
    plt.errorbar(x, r2_tr[:, p_idx], yerr=r2_tr_std[:, p_idx], 
                 label=f'NLLS R² {pname}', marker=markers['NLLS'], color=colors[pname], 
                 linestyle='--', capsize=4, capthick=1.5, linewidth=2, markersize=7)

plt.xlabel('Pyruvate SNR', fontsize=12)
plt.ylabel('Coefficient of Determination (R²)', fontsize=12)
plt.title('Pyruvate SNR vs R² (all parameters)', fontsize=13)
plt.legend(ncol=2, fontsize=10)
ax.set_ylim(-0.5, 1.05)
ax.grid(True, which='major', alpha=0.7)
ax.grid(True, which='minor', alpha=0.3, linestyle=':')
plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'SNR_vs_R2_all_params.png'), dpi=300, bbox_inches='tight')
plt.close()


# --- Combined plot: SNR vs R² for all parameters (NN and Traditional) ---
# Create a two-panel figure: left = Pyruvate SNR, right = Lactate SNR
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
colors = {'kPL': 'C0', 'kVE': 'C1', 'vB': 'C2'}
markers = {'NN': 'o', 'TR': 'x'}

snr_x = [snrs[:, 0], snrs[:, 1]]  # [pyruvate, lactate]
snr_labels = ['Pyruvate SNR', 'Lactate SNR']

for ax_idx, ax in enumerate(axes):
    x = snr_x[ax_idx]
    for p_idx, pname in enumerate(params):
        # NN - with error bars
        ax.errorbar(x, r2_nn[:, p_idx], yerr=r2_nn_std[:, p_idx],
                   label=f'NN R² {pname}', marker=markers['NN'], color=colors[pname], 
                   linestyle='-', capsize=3, capthick=1, linewidth=1.5, markersize=6)
        # Traditional - with error bars
        ax.errorbar(x, r2_tr[:, p_idx], yerr=r2_tr_std[:, p_idx],
                   label=f'TR R² {pname}', marker=markers['TR'], color=colors[pname], 
                   linestyle='--', capsize=3, capthick=1, linewidth=1.5, markersize=6)

    ax.set_xlabel(snr_labels[ax_idx])
    if ax_idx == 0:
        ax.set_ylabel('Coefficient of Determination (R²)')
    ax.set_title(f'{snr_labels[ax_idx]} vs R² (all parameters)')
    ax.set_ylim(-0.5, 1.05)
    ax.grid(True, which='major', alpha=0.7)
    ax.grid(True, which='minor', alpha=0.3, linestyle=':')

# Shared legend (avoid duplicate labels)
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(run_dir, 'SNR_vs_R2_all_params_twinx.png'), dpi=300, bbox_inches='tight')
plt.close()



plt.figure(figsize=(8, 6))
# ax = plt.gca()
fig, ax1 = plt.subplots()
colors = {'kPL': 'C0', 'kVE': 'C1', 'vB': 'C2'}
markers = {'NN': 'o', 'TR': 'x'}
for p_idx, pname in enumerate(params):
    x = snrs[:, 0]
    ax1.plot(x, r2_nn[:, p_idx], label=f'NN R² {pname}', marker=markers['NN'], color=colors[pname], linestyle='-')
    ax1.plot(x, r2_tr[:, p_idx], label=f'TR R² {pname}', marker=markers['TR'], color=colors[pname], linestyle='--')
    # Add shaded error bands if bootstrap std available
    if np.any(np.isfinite(r2_nn_std[:, p_idx])):
        ax1.fill_between(x, r2_nn[:, p_idx] - r2_nn_std[:, p_idx], r2_nn[:, p_idx] + r2_nn_std[:, p_idx],
                         color=colors[pname], alpha=0.15)
    if np.any(np.isfinite(r2_tr_std[:, p_idx])):
        ax1.fill_between(x, r2_tr[:, p_idx] - r2_tr_std[:, p_idx], r2_tr[:, p_idx] + r2_tr_std[:, p_idx],
                         color=colors[pname], alpha=0.08)

ax1.set_xlabel('Pyruvate SNR')
ax1.set_ylabel('Coefficient of Determination (R²)')
ax1.set_title('Pyruvate SNR vs R² (all parameters)')
# ax1.legend(ncol=2)
ax1.set_ylim(-0.5, 1.05)
ax1.grid(True, which='major', alpha=0.7)
ax1.grid(True, which='minor', alpha=0.3, linestyle=':')

ax2 = ax1.twiny()
ax2.set_xlabel('Lactate SNR')
# ax2.set_xlim(ax1.get_xlim())
for p_idx, pname in enumerate(params):
    x = snrs[:, 1]
    ax2.plot(x, r2_nn[:, p_idx], label=f'NN R² {pname}', marker=markers['NN'], color=colors[pname], linestyle='-')
    ax2.plot(x, r2_tr[:, p_idx], label=f'TR R² {pname}', marker=markers['TR'], color=colors[pname], linestyle='--')
    # Add shaded error bands if bootstrap std available
    if np.any(np.isfinite(r2_nn_std[:, p_idx])):
        ax2.fill_between(x, r2_nn[:, p_idx] - r2_nn_std[:, p_idx], r2_nn[:, p_idx] + r2_nn_std[:, p_idx],
                         color=colors[pname], alpha=0.15)
    if np.any(np.isfinite(r2_tr_std[:, p_idx])):
        ax2.fill_between(x, r2_tr[:, p_idx] - r2_tr_std[:, p_idx], r2_tr[:, p_idx] + r2_tr_std[:, p_idx],
                         color=colors[pname], alpha=0.08)


plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'SNR_vs_R2_all_params_twinAxis.png'), dpi=300, bbox_inches='tight')
plt.close()


# Calculate SNR for pyruvate and lactate signals
pyr_signals = X[:, :, 0]
lac_signals = X[:, :, 1]
snr_pyr = np.max(np.abs(pyr_signals), axis=1) / noise
snr_lac = np.max(np.abs(lac_signals), axis=1) / noise

print("\n--- Step 12: Generate Summary Report ---")
with open(os.path.join(run_dir, 'summary_report.md'), 'w') as f:
    f.write(f"# Hyperpolarized 13C MRI Analysis Demo Summary\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write(f"## Dataset\n")
    f.write(f"- Total samples: {n_samples}\n")
    f.write(f"- Test samples: {Xr_test.shape[0]}\n")
    f.write(f"- Time points: {n_timepoints} (TR=5s, 0-{(n_timepoints-1)*5}s)\n\n")
    
    f.write(f"## Parameter Ranges\n")
    f.write(f"- kPL: {generator.kpl_range[0]:.3f} - {generator.kpl_range[1]:.3f} s^-1\n")
    
    f.write(f"## Training\n")
    f.write(f"- Initial learning rate: 0.001\n")
    # f.write(f"- Traditional fitting time: {tradfit_time:.2f} seconds\n\n")
    
    f.write(f"Test R² for kpl: {r2_kpl:.3f}")
    f.write(f"Test R² for kve: {r2_kve:.3f}")
    f.write(f"Test R² for vb:  {r2_vb:.3f}")
    f.write(f"\n\n## SNR Analysis\n")
    f.write(f"- Mean SNR for Pyruvate: {np.mean(snr_pyr):.2f}\n")
    f.write(f"- Mean SNR for Lactate: {np.mean(snr_lac):.2f}\n")
    f.write(f"- Minimum SNR for Pyruvate: {np.min(snr_pyr):.2f}\n")
    f.write(f"- Minimum SNR for Lactate: {np.min(snr_lac):.2f}\n")
    f.write(f"- Maximum SNR for Pyruvate: {np.max(snr_pyr):.2f}\n")
    f.write(f"- Maximum SNR for Lactate: {np.max(snr_lac):.2f}\n")
    
        # Display statistics and plot comparison
    f.write("\n=== X_raw Statistics ===")
    f.write(f"Maximum: {np.max(X):.6f}\n")
    f.write(f"Minimum: {np.min(X):.6f}\n")
    f.write(f"Mean: {np.mean(X):.6f}\n")
    f.write(f"Shape: {X.shape}\n")

    f.write("\n=== X_norm Statistics ===")
    f.write(f"Maximum: {np.max(X_norm):.6f}\n")
    f.write(f"Minimum: {np.min(X_norm):.6f}\n")
    f.write(f"Mean: {np.mean(X_norm):.6f}\n")
    f.write(f"Shape: {X_norm.shape}\n")
    
    



