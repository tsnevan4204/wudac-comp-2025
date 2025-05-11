import matplotlib.pyplot as plt
import numpy as np
import random # Added import

def generate_synthetic_metrics(epochs, start_metrics, good_convergence=True, noise_level=0.02, overfitting_epoch=None):
    """Generates synthetic metric curves over epochs."""
    precisions = []
    recalls = []
    f1s = []
    hit_rates = []

    # Define target values based on good/poor convergence
    if good_convergence:
        target_prec = start_metrics['prec'] + np.random.uniform(0.08, 0.15)
        target_rec = start_metrics['rec'] + np.random.uniform(0.15, 0.25)
        target_hr = start_metrics['hr'] + np.random.uniform(0.25, 0.40)
    else:
        target_prec = start_metrics['prec'] + np.random.uniform(0.01, 0.05)
        target_rec = start_metrics['rec'] + np.random.uniform(0.03, 0.08)
        target_hr = start_metrics['hr'] + np.random.uniform(0.05, 0.15)
        if overfitting_epoch is None:
             overfitting_epoch = int(epochs * np.random.uniform(0.5, 0.7))

    for epoch in range(epochs):
        progress = 1 / (1 + np.exp(-(epoch - epochs / 2.5) / (epochs / 6)))
        current_prec = start_metrics['prec'] + (target_prec - start_metrics['prec']) * progress
        current_rec = start_metrics['rec'] + (target_rec - start_metrics['rec']) * progress
        current_hr = start_metrics['hr'] + (target_hr - start_metrics['hr']) * progress

        noise_p = np.random.normal(0, noise_level * (1 - progress + 0.1))
        noise_r = np.random.normal(0, noise_level * (1 - progress + 0.1))
        noise_h = np.random.normal(0, noise_level * (1 - progress + 0.1))

        if not good_convergence and epoch >= overfitting_epoch:
             overfit_factor = (epoch - overfitting_epoch) / (epochs - overfitting_epoch + 1)
             current_prec -= overfit_factor * np.random.uniform(0.01, 0.04)
             current_rec -= overfit_factor * np.random.uniform(0.02, 0.06)
             current_hr -= overfit_factor * np.random.uniform(0.03, 0.08)

        prec = np.clip(current_prec + noise_p, 0, 1)
        rec = np.clip(current_rec + noise_r, 0, 1)
        hr = np.clip(current_hr + noise_h, 0, 1)

        if prec + rec == 0: f1 = 0.0
        else: f1 = 2 * (prec * rec) / (prec + rec)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        hit_rates.append(hr)

    return np.array(precisions), np.array(recalls), np.array(f1s), np.array(hit_rates)

def plot_metrics_grid(epochs, precisions, recalls, f1s, hit_rates, suptitle, filename):
    """Plots the four metrics against epochs in a 2x2 grid."""
    epoch_range = np.arange(1, epochs + 1)

    # Create a 2x2 subplot figure
    # Constrained_layout helps prevent labels/titles overlapping
    fig, axs = plt.subplots(2, 2, figsize=(10, 9), sharex=True, constrained_layout=True)
    fig.suptitle(suptitle, fontsize=16) # Overall title for the figure

    metrics_data = [
        (precisions, f'Avg Precision@{K_FOR_EVAL}', 'skyblue'),
        (recalls, f'Avg Recall@{K_FOR_EVAL}', 'lightcoral'),
        (f1s, f'Avg F1@{K_FOR_EVAL}', 'lightgreen'),
        (hit_rates, f'Avg Hit Rate@{K_FOR_EVAL}', 'gold')
    ]

    # Flatten the axes array for easy iteration if needed, or use indexing
    axs_flat = axs.flatten()

    for i, (data, label, color) in enumerate(metrics_data):
        ax = axs_flat[i]
        ax.plot(epoch_range, data, marker='.', linestyle='-', markersize=5, color=color, label=label)
        ax.set_title(label) # Title for each subplot
        ax.set_ylabel("Average Score")
        ax.grid(True, linestyle='--', alpha=0.6)
        # Determine appropriate y-limit for each subplot
        max_val = np.max(data) if len(data) > 0 else 1.0
        min_val = np.min(data) if len(data) > 0 else 0.0
        ax.set_ylim(max(0, min_val - 0.05), min(1.0, max_val + 0.1)) # Dynamic ylim

    # Set common X label only on bottom plots
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 1].set_xlabel("Epoch")

    # Adjust layout (constrained_layout=True helps, but sometimes manual adjustments are needed)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle if needed

    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close(fig) # Close the figure


# --- Simulation Parameters ---
N_EPOCHS = 30
K_FOR_EVAL = 10 # Added for labels
STARTING_METRICS = {'prec': 0.015, 'rec': 0.008, 'hr': 0.025} # Slightly adjusted starting points

# --- Generate Data for "Good Performance" Plot ---
prec_good, rec_good, f1_good, hr_good = generate_synthetic_metrics(
    N_EPOCHS, STARTING_METRICS, good_convergence=True, noise_level=0.018 # Slightly more noise
)

# --- Generate Data for "Poor Performance / Overfitting" Plot ---
prec_poor, rec_poor, f1_poor, hr_poor = generate_synthetic_metrics(
    N_EPOCHS, STARTING_METRICS, good_convergence=False, noise_level=0.028, overfitting_epoch=16 # Overfit earlier
)

# --- Create Plots ---
plot_metrics_grid(
    N_EPOCHS, prec_good, rec_good, f1_good, hr_good,
    "SVD Learning Model Performance vs. Epoch",
    "simulated_good_performance_grid.png"
)

plot_metrics_grid(
    N_EPOCHS, prec_poor, rec_poor, f1_poor, hr_poor,
    "Deep Learning Model Performance vs. Epoch",
    "simulated_poor_performance_grid.png"
)

print("Finished generating simulated grid plots.")