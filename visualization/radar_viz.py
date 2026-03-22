import matplotlib.pyplot as plt
import numpy as np


def plot_radar_returns(time_steps, true_pos, radar_pos, blackout_mask, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    x_true = true_pos[:, 0]
    z_true = true_pos[:, 2]

    x_radar = radar_pos[:, 0]
    z_radar = radar_pos[:, 2]

    ax.plot(x_true, z_true, "b-", label="True Trajectory", linewidth=2)

    valid_idx = ~blackout_mask
    ax.scatter(
        x_radar[valid_idx],
        z_radar[valid_idx],
        c="g",
        marker="+",
        label="Radar Returns",
        s=50,
        alpha=0.7,
    )

    if np.any(blackout_mask):
        blackout_start = np.where(blackout_mask)[0][0]
        blackout_end = np.where(blackout_mask)[0][-1]

        ax.axvspan(
            x_true[blackout_start],
            x_true[blackout_end],
            color="red",
            alpha=0.2,
            label="Plasma Blackout Region",
        )

    ax.set_xlabel("Downrange Distance (m)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Radar Measurements Under Plasma Blackout")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

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

    labeler = BlackoutLabeler()
    mask = labeler.label_trajectory(traj["plasma_profile"])

    time_steps = np.arange(len(mask)) * 0.1

    os.makedirs("assets/results", exist_ok=True)
    plot_radar_returns(
        time_steps,
        traj["trajectory"],
        traj["radar_returns"],
        mask,
        save_path="assets/results/radar_viz.png",
    )
    print("Radar returns plot saved.")
