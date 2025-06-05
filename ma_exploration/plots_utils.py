import numpy as np
import matplotlib.pyplot as plt

def plot_u_histograms(U_noisy, costs, lambda_, control_names=None):
    """
    Plots weighted histograms for each control dimension.
    
    Args:
        U_noisy: Noisy control samples (shape: num_samples x horizon x control_dim).
        costs: Cost array (shape: num_samples,).
        lambda_: MPPI temperature parameter.
        control_names: List of names for each control dimension (e.g., ["v", "w"]).
    """
    num_samples, horizon, control_dim = U_noisy.shape
    weights = np.exp(-(costs - np.min(costs)) / lambda_)
    
    # Default names: u1, u2, ...
    if control_names is None:
        control_names = [f'u{i+1}' for i in range(control_dim)]
    
    # Create subplots
    fig, axes = plt.subplots(1, control_dim, figsize=(5 * control_dim, 4))
    if control_dim == 1:
        axes = [axes]  # Ensure axes is iterable for single-dimension cases
    
    for i, (ax, name) in enumerate(zip(axes, control_names)):
        # Flatten all samples for the i-th control dimension
        ui_samples = U_noisy[:, :, i].flatten()
        
        # Plot weighted histogram
        ax.hist(ui_samples, bins=50, weights=np.repeat(weights, horizon), 
                alpha=0.7, color=f'C{i}', edgecolor='k')
        ax.set_xlabel(f'Control {name}')
        ax.set_ylabel('Weighted Frequency')
        ax.set_title(f'Weight Distribution for {name}')
    
    plt.tight_layout()
    plt.show()