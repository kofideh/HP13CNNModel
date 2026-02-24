import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from hybrid_model_utils import (
    HybridMultiHead,
    VB_MIN, VB_MAX, KPL_MIN, KPL_MAX, KVE_MIN, KVE_MAX
)
from two_compartment_generator import (
    TwoCompartmentHPDataGenerator
)   
from two_compartment_generator_measured import TwoCompartmentHPDataGeneratorMeasured
    

AIF_TYPE = 'Measured'  # 'measured' or 'GAMMA-Variate'
# AIF_TYPE = 'GAMMA-Variate'  # 'measured' or 'GAMMA-Variate'

#for brain data from doi: 10.1002/hbm.26329
# NUM_TIMEPOINTS = 12
# PYR_FA_SCHEDULE = [11.0] * NUM_TIMEPOINTS 
# LAC_FA_SCHEDULE = [20.0] * NUM_TIMEPOINTS 
# SCAN_TR = 5.0  # seconds

#for TRAMP Mouse data from doi: 10.1002/mrm.2612
# NUM_TIMEPOINTS = 16
# DEFAULT_VFA_SCHEDULE = np.array([
#     14.4775, 14.9632, 15.5014, 16.1021, 16.7787, 17.5484, 18.4349, 19.4712,
#     20.7048, 22.2077, 24.0948, 26.5651, 30.0000, 35.2644, 45.0000, 90.0000
# ], dtype=float)

# PYR_FA_SCHEDULE = DEFAULT_VFA_SCHEDULE
# LAC_FA_SCHEDULE = DEFAULT_VFA_SCHEDULE
# SCAN_TR = 2.0  # seconds

# #for rat kidney data from doi: 10.1002/mrm.2612
NUM_TIMEPOINTS = 25
PYR_FA_SCHEDULE = [15] * NUM_TIMEPOINTS 
LAC_FA_SCHEDULE = [15] * NUM_TIMEPOINTS 
SCAN_TR = 2.0  # seconds
KPL_MAX = 0.06  # s^-1, reduced from 0.20 to better match the rat kidney data range


n_samples = 1000000
n_epochs = 1000
noise_level = 1/(20*1)  # Standard deviation of noise to add

# 1. Generate data
time_points = np.arange(0, len(PYR_FA_SCHEDULE) * SCAN_TR, SCAN_TR)
n_timepoints = len(time_points)


if AIF_TYPE == 'GAMMA-Variate':
    generator = TwoCompartmentHPDataGenerator(
        vb_range=(VB_MIN, VB_MAX),
        kpl_range=(KPL_MIN, KPL_MAX),
        kve_range=(KVE_MIN, KVE_MAX),
        r1p_range=(1/30, 1/30),
        r1l_range=(1/25, 1/25),
        time_points=None,                        # inferred from schedule length & TR
        flip_angle_pyr_deg=PYR_FA_SCHEDULE, # or scalar FAs
        flip_angle_lac_deg=LAC_FA_SCHEDULE,
        TR=SCAN_TR
    )
else:   
    generator = TwoCompartmentHPDataGeneratorMeasured(
    vb_range=(VB_MIN, VB_MAX),
    kpl_range=(KPL_MIN, KPL_MAX),
    kve_range=(KVE_MIN, KVE_MAX),
    r1p_range=(1/30, 1/30),
    r1l_range=(1/25, 1/25),
    time_points=None,                        # inferred from schedule length & TR
    flip_angle_pyr_deg=PYR_FA_SCHEDULE, # or scalar FAs
    flip_angle_lac_deg=LAC_FA_SCHEDULE,
    TR=SCAN_TR
)


X, y = generator.generate_dataset(n_samples=n_samples, noise_std=noise_level)

# 2. Split channels
X_raw = X.copy()  
X_norm = X 



output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join(output_dir, f"noiselevel_{noise_level}_{timestamp}")
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
X_raw_flat = X_raw.reshape(X.shape[0], -1)
X_norm_flat = X_norm.reshape(X.shape[0], -1)

data_peak = X_raw_flat.max()


Xr_train, Xr_temp, Xn_train, Xn_temp, y_train, y_temp = train_test_split(
    X_raw_flat, X_norm_flat, y, test_size=0.3, random_state=42)
Xr_val, Xr_test, Xn_val, Xn_test, y_val, y_test = train_test_split(
    Xr_temp, Xn_temp, y_temp, test_size=0.05, random_state=42)


# 5. Convert to tensors
Xr_train_tensor = torch.tensor(Xr_train, dtype=torch.float32)
Xn_train_tensor = torch.tensor(Xn_train, dtype=torch.float32)
Xr_val_tensor = torch.tensor(Xr_val, dtype=torch.float32)
Xn_val_tensor = torch.tensor(Xn_val, dtype=torch.float32)
Xr_test_tensor = torch.tensor(Xr_test, dtype=torch.float32)
Xn_test_tensor = torch.tensor(Xn_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


model = HybridMultiHead(input_dim_raw=X_raw_flat.shape[1], input_dim_norm=X_norm_flat.shape[1])


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



# Early stopping parameters
patience = 50  # Number of epochs to wait for improvement
min_delta = 1e-6  # Minimum change to qualify as an improvement
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

# 7. Training loop with early stopping
print("Training Neural Network with Early Stopping...")
import time
training_start = time.perf_counter()
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    preds = model(Xn_train_tensor, Xr_train_tensor)
    loss = criterion(preds, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(Xn_val_tensor, Xr_val_tensor)
        val_loss = criterion(val_preds, y_val_tensor)
    
    # Store losses for plotting
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    
    # Early stopping logic
    if val_loss.item() < best_val_loss - min_delta:
        best_val_loss = val_loss.item()
        patience_counter = 0
        # Save best model state
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}, Best Val Loss = {best_val_loss:.4f}, Patience = {patience_counter}/{patience}")
    
    # Check if we should stop early
    if patience_counter >= patience:
        print(f"Early stopping triggered after epoch {epoch}! Best validation loss: {best_val_loss:.4f}")
        break

training_duration = time.perf_counter() - training_start
print(f"Training finished in {training_duration:.2f} seconds")

# Load best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

# Plot training curves
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', alpha=0.7)
plt.plot(val_losses, label='Validation Loss', alpha=0.7)
plt.axvline(x=len(train_losses) - patience_counter - 1, color='red', linestyle='--', 
            label=f'Best Model (Epoch {len(train_losses) - patience_counter - 1})', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale for better visualization
plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
plt.close()

model_path = os.path.join(run_dir, "trained_hybrid_positive.pth")
if best_model_state is not None:
    model.load_state_dict(best_model_state)
torch.save(model.state_dict(), model_path)
# After training
# torch.save(model.state_dict(), "trained_hybrid_positive.pth")
print("Saved model with positivity constraints to trained_hybrid_positive.pth")
# 8. Evaluation
model.eval()


r2_kpl = 0
r2_kve = 0
r2_vb  = 0


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

    plot_true_vs_pred(y_true, y_pred, f"{noise_level}_Neural Network")


    
    # Step 12: Generate a summary report
# Estimate SNR
pyr_signals = X[:, :, 0]
lac_signals = X[:, :, 1]

# Compute SNR as mean amplitude divided by noise std
snr_pyr = np.max(np.abs(pyr_signals), axis=1) / noise_level
snr_lac = np.max(np.abs(lac_signals), axis=1) / noise_level

print("\n--- Step 12: Generate Summary Report ---")
with open(os.path.join(run_dir, 'training_report.md'), 'w') as f:
    f.write(f"# Hyperpolarized 13C MRI Analysis Demo Summary\n\n")
    f.write(f"**Generator Name:** {generator.__class__.__name__}\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write(f"## Dataset\n")
    f.write(f"Total samples: {n_samples}\n")
    f.write(f"Training samples: {Xr_train.shape[0]}\n")
    f.write(f"Validation samples: {Xr_val.shape[0]}\n")
    f.write(f"Test samples: {Xr_test.shape[0]}\n")
    f.write(f"Noise level (std): {noise_level}\n")
    
    f.write(f"## Configuration\n")
    f.write(f"NUM_TIME_POINTS: {n_timepoints}\n")
    f.write(f"SCAN_TR={SCAN_TR}\n")
    f.write(f"PYR_FA_SCHEDULE: {PYR_FA_SCHEDULE}\n")
    f.write(f"LAC_FA_SCHEDULE: {LAC_FA_SCHEDULE}\n")
    f.write(f"TRAINING_PEAK: {np.max(X):.6f}\n")# include the training peak value used for normalization
    f.write(f"## Parameter Ranges\n")
    f.write(f"kPL Range: {KPL_MIN:.3f} - {KPL_MAX:.3f} s^-1\n")
    f.write(f"kVE Range: {KVE_MIN:.3f} - {KVE_MAX:.3f} s^-1\n")
    f.write(f"vB Range: {VB_MIN:.3f} - {VB_MAX:.3f}\n\n")
    
    f.write(f"## Training\n")
    f.write(f"Max epochs: {n_epochs}\n")
    f.write(f"Actual epochs trained: {len(train_losses)}\n")
    f.write(f"Early stopping patience: {patience}\n")
    f.write(f"Early stopping triggered: {'Yes' if patience_counter >= patience else 'No'}\n")
    f.write(f"Best validation loss: {best_val_loss:.6f}\n")
    f.write(f"Final training loss: {train_losses[-1]:.6f}\n")
    f.write(f"Initial learning rate: 0.001\n")
    f.write(f"Training time: {training_duration:.2f} seconds\n")
    # f.write(f"- Traditional fitting time: {tradfit_time:.2f} seconds\n\n")
    
    f.write(f"Test R² for kpl: {r2_kpl:.3f}")
    f.write(f"Test R² for kve: {r2_kve:.3f}")
    f.write(f"Test R² for vb:  {r2_vb:.3f}")
    f.write(f"\n\n## SNR Analysis\n")
    f.write(f"Mean SNR for Pyruvate: {np.mean(snr_pyr):.2f}\n")
    f.write(f"Mean SNR for Lactate: {np.mean(snr_lac):.2f}\n")
    f.write(f"Minimum SNR for Pyruvate: {np.min(snr_pyr):.2f}\n")
    f.write(f"Minimum SNR for Lactate: {np.min(snr_lac):.2f}\n")
    f.write(f"Maximum SNR for Pyruvate: {np.max(snr_pyr):.2f}\n")
    f.write(f"Maximum SNR for Lactate: {np.max(snr_lac):.2f}\n")
    
        # Display statistics and plot comparison
    f.write("\n=== X_raw Statistics ===\n")
    f.write(f"Maximum: {np.max(X):.6f}\n")
    f.write(f"Minimum: {np.min(X):.6f}\n")
    f.write(f"Mean: {np.mean(X):.6f}\n")
    f.write(f"Shape: {X.shape}\n")

    f.write("\n=== X_norm Statistics ===\n")
    f.write(f"Maximum: {np.max(X_norm):.6f}\n")
    f.write(f"Minimum: {np.min(X_norm):.6f}\n")
    f.write(f"Mean: {np.mean(X_norm):.6f}\n")
    f.write(f"Shape: {X_norm.shape}\n")
    

    

    

    
    



