import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import matplotlib.colors as mcolors

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

def plot_rollouts_and_hist(state_rollouts,          # (M, K+1, state_dim)
                           weights,                 # (M,)
                           goal=None,
                           cmap_name="viridis",
                           sort_hist=True,
                           optimal_rollout=None):   # (1, K+1, state_dim)
    """
    • Left: trajectories coloured & thickened by weight, with a colour-bar.
    • Right: histogram of weights, bars coloured with the same mapping.
    • Optional: draws optimal rollout in bold red.
    """
    M, Kp1, _ = state_rollouts.shape
    pos = state_rollouts[:, :, :2]

    # Normalize weights → [0,1]
    w_norm = (weights - weights.min()) / (weights.ptp() + 1e-9)
    cmap   = cm.get_cmap(cmap_name)
    smap   = cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap)

    fig, (ax_traj, ax_hist) = plt.subplots(1, 2, figsize=(10, 5),
                                           gridspec_kw={"width_ratios": [2, 1]})

    # ─────────────── Rollout Trajectories ───────────────
    for i in range(M):
        colour = cmap(w_norm[i])
        lw     = 0.5 + 3.0 * w_norm[i]
        ax_traj.plot(pos[i, :, 0], pos[i, :, 1],
                     color=colour, linewidth=lw)

    # Plot optimal rollout
    if optimal_rollout is not None:
        best_xy = optimal_rollout[0, :, :2]
        ax_traj.plot(best_xy[:, 0], best_xy[:, 1],
                     color='red', linewidth=3.0, label='Best rollout')

    if goal is not None:
        ax_traj.plot(goal[0], goal[1], 'g*', markersize=12, label="goal")

    ax_traj.set_aspect("equal", "box")
    ax_traj.set_title("Rollouts coloured by weight")
    ax_traj.legend(loc="upper right")
    fig.colorbar(smap, ax=ax_traj, label="normalised weight")

    # ─────────────── Histogram ───────────────
    order = np.argsort(weights) if sort_hist else np.arange(M)
    bar_colors = cmap(w_norm[order])
    ax_hist.bar(np.arange(M), weights[order], color=bar_colors)
    ax_hist.set_title("Weight distribution")
    ax_hist.set_xlabel("sample index (sorted)" if sort_hist else "sample index")
    ax_hist.set_ylabel("weight")

    plt.tight_layout()
    plt.show()



def plot_rollouts_with_collision(rollouts, weights, collided_mask, goal=None, optimal_rollout=None):
    """
    Plots all rollouts with collision info.
    Dashed lines indicate collisions.
    Colors based on weights.
    Best rollout (if given) is drawn in red.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    norm = mcolors.Normalize(vmin=np.min(weights), vmax=np.max(weights))
    cmap = cm.get_cmap('viridis')

    for i, traj in enumerate(rollouts):
        color = cmap(norm(weights[i]))
        linestyle = '--' if collided_mask[i] else '-'
        ax.plot(traj[:, 0], traj[:, 1], color=color, linestyle=linestyle, alpha=0.7)

    # Plot optimal rollout
    if optimal_rollout is not None:
        best_xy = optimal_rollout[0, :, :2]
        ax.plot(best_xy[:, 0], best_xy[:, 1],
                color='red', linewidth=3.0, label='Best rollout')

    if goal is not None:
        ax.plot(goal[0], goal[1], 'g*', markersize=15, label="Goal")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="MPPI Weights")

    ax.set_aspect('equal')
    ax.set_title("Rollouts with Collision (dashed) and Weight Coloring")
    ax.legend(loc='upper right')
    plt.grid(True)
    plt.show()
