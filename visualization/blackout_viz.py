import matplotlib.pyplot as plt
import numpy as np


def plot_blackout_profile(time_steps, ne_profile, threshold=1e18, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogy(time_steps, ne_profile, label="Electron Density (Ne)", color="b", linewidth=2)
    ax.axhline(threshold, color="r", linestyle="--", label="Blackout Threshold")

    ax.fill_between(
        time_steps,
        1e12,
        ne_profile,
        where=(ne_profile >= threshold),
        color="red",
        alpha=0.3,
        label="Blackout Region",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Electron Density (m⁻³)")
    ax.set_title("Plasma Sheath Electron Density Profile")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    from simulation.trajectory_gen import TrajectoryGenerator
    from preprocessing.blackout_labeler import BlackoutLabeler
    import os

    gen = TrajectoryGenerator()
    traj = gen.generate_batch(1)[0]

    ne_profile = traj["plasma_profile"]
    time_steps = np.arange(len(ne_profile)) * 0.1

    os.makedirs("assets/results", exist_ok=True)
    plot_blackout_profile(time_steps, ne_profile, save_path="assets/results/blackout_profile.png")
    print("Blackout profile plot saved.")
