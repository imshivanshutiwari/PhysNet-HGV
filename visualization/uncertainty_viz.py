import matplotlib.pyplot as plt
import numpy as np


def plot_uncertainty(
    time_steps, errors, covariances, title="Tracking Error & Uncertainty (±3σ)", save_path=None
):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    labels = ["X Position", "Y Position", "Z Position"]

    for i in range(3):
        err = errors[:, i]

        sigma3 = 3 * np.sqrt(np.abs(covariances[:, i, i]))

        axs[i].plot(time_steps, err, "b-", label="Error")
        axs[i].plot(time_steps, sigma3, "r--", label="+3σ Bound")
        axs[i].plot(time_steps, -sigma3, "r--", label="-3σ Bound")
        axs[i].fill_between(time_steps, -sigma3, sigma3, color="r", alpha=0.1)

        axs[i].set_ylabel(f"{labels[i]} Error (m)")
        axs[i].legend(loc="upper right")
        axs[i].grid(True, linestyle="--", alpha=0.7)

    axs[2].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    N = 100
    time_steps = np.arange(N) * 0.1
    errors = np.random.randn(N, 3) * 5.0

    covs = np.zeros((N, 3, 3))
    for i in range(N):
        if 30 < i < 70:
            covs[i] = np.eye(3) * ((i - 30) * 1.5) ** 2 + np.eye(3) * 25
        else:
            covs[i] = np.eye(3) * 25

    import os

    os.makedirs("assets/results", exist_ok=True)
    plot_uncertainty(time_steps, errors, covs, save_path="assets/results/uncertainty_viz.png")
    print("Uncertainty plot saved.")
