
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pathlib import Path

# KPL_MIN, KPL_MAX = 0.01, 0.20
# KVE_MIN, KVE_MAX = 0.05, 0.45
# VB_MIN,  VB_MAX  = 0.01, 0.50

KPL_MIN, KPL_MAX = 0.001, 0.20
KVE_MIN, KVE_MAX = 0.05, 0.45
VB_MIN,  VB_MAX =  0.03, 0.18

class HybridMultiHead(nn.Module):
    def __init__(self, input_dim_raw, input_dim_norm):
        super(HybridMultiHead, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim_norm, 128),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Add constraints for physical parameters
        self.kpl_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()  # Ensure positive kpl values
        )
        self.kve_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()  # Ensure positive kve values
        )
        
        self.vb_head = nn.Sequential(
            nn.Linear(64 + input_dim_raw, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


    def forward(self, x_norm, x_raw):
        shared_out = self.shared(x_norm)
        kpl_hat = self.kpl_head(shared_out)      # in (0,1) after Sigmoid
        kve_hat = self.kve_head(shared_out)      # in (0,1)
        vb_hat = self.vb_head(torch.cat([shared_out, x_raw], dim=1))# in (0,1)

        kpl = KPL_MIN + kpl_hat * (KPL_MAX - KPL_MIN)
        kve = KVE_MIN + kve_hat * (KVE_MAX - KVE_MIN)
        vb  = VB_MIN  + vb_hat  * (VB_MAX  - VB_MIN)
        return torch.cat([kpl, kve, vb], dim=1)


def load_nifti_series(filepaths):
    data = [nib.load(str(p)).get_fdata() for p in filepaths]
    return np.stack(data, axis=-1)

def preprocess_signals(raw_signals):
    # raw_signals: (samples, timepoints, 2)
    # norm = raw_signals / (np.max(raw_signals, axis=1, keepdims=True) + 1e-6)
    norm = raw_signals/np.max(raw_signals)  # Normalize by global max
    raw_flat = raw_signals.reshape(raw_signals.shape[0], -1)
    norm_flat = norm.reshape(norm.shape[0], -1)
    return norm_flat.astype(np.float32), raw_flat.astype(np.float32)

def reshape_signals(raw_signals):
    # raw_signals: (samples, timepoints, 2)
    norm = raw_signals / (np.max(raw_signals, axis=1, keepdims=True) + 1e-6)
    raw_flat = raw_signals.reshape(raw_signals.shape[0], -1)
    norm_flat = norm.reshape(norm.shape[0], -1)  # Fix: use norm instead of raw_signals
    return norm_flat.astype(np.float32), raw_flat.astype(np.float32)

def evaluate_model(model, x_norm, x_raw, y_true=None):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_norm), torch.tensor(x_raw)).numpy()
    if y_true is not None:
        r2 = r2_score(y_true, pred, multioutput='raw_values')
        print(f"R² scores — kpl: {r2[0]:.3f}, kve: {r2[1]:.3f}, vb: {r2[2]:.3f}")
    return pred

def plot_prediction_scatter(y_true, y_pred, labels=["kpl", "kve", "vb"], saveName=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 4))
    for i, label in enumerate(labels):
        plt.subplot(1, 3, i+1)
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
        plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                 [y_true[:, i].min(), y_true[:, i].max()], 'r--')
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        plt.title(f"{label} (R²={r2:.2f})")
        plt.xlabel("True")
        plt.ylabel("Predicted")
    plt.tight_layout()
    if saveName:
        plt.savefig(saveName)
    else:
        plt.show()
    plt.close()

def plot_weight_histograms(model):
    plt.figure(figsize=(10, 5))
    for i, (name, param) in enumerate(model.named_parameters()):
        if "weight" in name and param.requires_grad:
            plt.subplot(2, 3, i+1)
            plt.hist(param.detach().cpu().numpy().flatten(), bins=30)
            plt.title(name)
    plt.tight_layout()
    plt.show()
    
    
    

# def plot_prediction_scatter2(y_true, y_pred, labels=["kpl", "kve", "vb"], saveName=None):
#     """
#     Generates scatter plots of true vs. predicted values for multiple targets,
#     after removing samples with NaN values in either y_true or y_pred for each target.

#     Args:
#         y_true (np.ndarray): Array of true target values, shape (n_samples, n_targets).
#         y_pred (np.ndarray): Array of predicted target values, shape (n_samples, n_targets).
#         labels (list, optional): List of string labels for each target.
#                                  Defaults to ["kpl", "kve", "vb"].
#         saveName (str, optional): If provided, the plot will be saved to this filename.
#                                   Otherwise, the plot will be shown. Defaults to None.
#     """
#     if not isinstance(y_true, np.ndarray):
#         y_true = np.array(y_true)
#     if not isinstance(y_pred, np.ndarray):
#         y_pred = np.array(y_pred)

#     if y_true.shape != y_pred.shape:
#         raise ValueError("y_true and y_pred must have the same shape.")
#     if y_true.ndim == 1: # If 1D, reshape to 2D for consistency
#         y_true = y_true.reshape(-1, 1)
#         y_pred = y_pred.reshape(-1, 1)
#     if len(labels) != y_true.shape[1]:
#         raise ValueError("Length of labels must match the number of targets (columns in y_true/y_pred).")

    
#     plt.figure(figsize=(5 * len(labels), 4)) # Adjust figure size based on number of labels


#     fontsize = 20
#     # plt.rcParams.update({'font.size': fontsize, 'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
#     #                      'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize,
#     #                      'legend.fontsize': fontsize, 'figure.titlesize': fontsize})
#     plt.rcParams.update({'font.size': fontsize, 'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
#                          'legend.fontsize': fontsize})
#     for i, label in enumerate(labels):
#         plt.subplot(1, len(labels), i + 1)

#         # Extract current column for true and predicted values
#         y_true_col = y_true[:, i]
#         y_pred_col = y_pred[:, i]

#         # Create a mask for non-NaN values in both y_true_col and y_pred_col
#         # A value is True if BOTH y_true_col and y_pred_col at that index are not NaN
#         valid_mask = ~np.isnan(y_true_col) & ~np.isnan(y_pred_col)

#         # Apply the mask to get cleaned data
#         y_true_cleaned = y_true_col[valid_mask]
#         y_pred_cleaned = y_pred_col[valid_mask]

#         if len(y_true_cleaned) < 2 or len(y_pred_cleaned) < 2:
#             # Not enough data points to plot or calculate R2 score reliably
#             plt.title(f"{label} (Not enough data)")
#             plt.xlabel("True")
#             plt.ylabel("Predicted")
#             plt.text(0.5, 0.5, 'Insufficient non-NaN data',
#                      horizontalalignment='center',
#                      verticalalignment='center',
#                      transform=plt.gca().transAxes)
#             print(f"Skipping plot for '{label}' due to insufficient non-NaN data points after cleaning.")
#             continue # Skip to the next label

#         # Proceed with plotting and R2 calculation using cleaned data
#         plt.scatter(y_true_cleaned, y_pred_cleaned, alpha=0.6, label='Predictions')

#         # Plot the y=x line (perfect prediction line)
#         # Use the min/max of the cleaned true values for the line range
#         min_val = y_true_cleaned.min()
#         max_val = y_true_cleaned.max()
#         plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y=x)')

#         # Calculate R2 score using cleaned data
#         r2 = r2_score(y_true_cleaned, y_pred_cleaned)
#         plt.title(f"{label} (R²={r2:.2f})", fontsize=12)
#         plt.xlabel("True Values")
#         plt.ylabel("Predicted Values")
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.legend()

#     plt.tight_layout()

#     if saveName:
#         try:
#             plt.savefig(saveName, dpi=300, bbox_inches='tight')
#             print(f"Plot saved to {saveName}")
#         except Exception as e:
#             print(f"Error saving plot: {e}")
#     else:
#         plt.show()

#     plt.close() # Close the figure to free up memory



def plot_prediction_scatter2(y_true, y_pred, labels=["kpl", "kve", "vb"], saveName=None):
    """
    Generates scatter plots of true vs. predicted values for multiple targets,
    after removing samples with NaN values in either y_true or y_pred for each target.

    Args:
        y_true (np.ndarray): Array of true target values, shape (n_samples, n_targets).
        y_pred (np.ndarray): Array of predicted target values, shape (n_samples, n_targets).
        labels (list, optional): List of string labels for each target.
                                 Defaults to ["kpl", "kve", "vb"].
        saveName (str, optional): If provided, the plot will be saved to this filename.
                                  Otherwise, the plot will be shown. Defaults to None.
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.ndim == 1: # If 1D, reshape to 2D for consistency
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    if len(labels) != y_true.shape[1]:
        raise ValueError("Length of labels must match the number of targets (columns in y_true/y_pred).")

    
    plt.figure(figsize=(5 * len(labels), 4)) # Adjust figure size based on number of labels


    fontsize = 20
    # plt.rcParams.update({'font.size': fontsize, 'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
    #                      'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize,
    #                      'legend.fontsize': fontsize, 'figure.titlesize': fontsize})
    plt.rcParams.update({'font.size': fontsize, 'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
                         'legend.fontsize': fontsize})
    for i, label in enumerate(labels):
        plt.subplot(1, len(labels), i + 1)

        # Extract current column for true and predicted values
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]

        # Create a mask for non-NaN values in both y_true_col and y_pred_col
        # A value is True if BOTH y_true_col and y_pred_col at that index are not NaN
        valid_mask = ~np.isnan(y_true_col) & ~np.isnan(y_pred_col)

        # Apply the mask to get cleaned data
        y_true_cleaned = y_true_col[valid_mask]
        y_pred_cleaned = y_pred_col[valid_mask]

        if len(y_true_cleaned) < 2 or len(y_pred_cleaned) < 2:
            # Not enough data points to plot or calculate R2 score reliably
            # plt.title(f"{label} (Not enough data)")
            # plt.xlabel("True")
            # plt.ylabel("Predicted")
            # plt.text(0.5, 0.5, 'Insufficient non-NaN data',
            #          horizontalalignment='center',
            #          verticalalignment='center',
            #          transform=plt.gca().transAxes)
            # print(f"Skipping plot for '{label}' due to insufficient non-NaN data points after cleaning.")
            # continue # Skip to the next label
            y_true_cleaned = y_true_col
            y_pred_cleaned = y_pred_col

        # Proceed with plotting and R2 calculation using cleaned data
        plt.scatter(y_true_cleaned, y_pred_cleaned, alpha=0.6, label='Predictions')

        # Plot the y=x line (perfect prediction line)
        # Use the min/max of the cleaned true values for the line range
        min_val = y_true_cleaned.min()
        max_val = y_true_cleaned.max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y=x)')

        # Calculate R2 score using cleaned data
        if len(y_true_col[valid_mask]) < 2 or len(y_pred_col[valid_mask]) < 2:
            plt.title(f"{label}", fontsize=12)
        else:
            r2 = r2_score(y_true_cleaned, y_pred_cleaned)
            plt.title(f"{label} (R²={r2:.2f})", fontsize=12)
            
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

    plt.tight_layout()

    if saveName:
        try:
            plt.savefig(saveName, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {saveName}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        plt.show()

    plt.close() # Close the figure to free up memory