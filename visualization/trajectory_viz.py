import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_3d(
    true_trajectory, estimated_trajectory=None, blackout_mask=None, save_path=None
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        true_trajectory[:, 0],
        true_trajectory[:, 1],
        true_trajectory[:, 2],
        label="True Trajectory",
        color="blue",
        linewidth=2,
    )

    if blackout_mask is not None:
        blackout_points = true_trajectory[blackout_mask]
        ax.scatter(
            blackout_points[:, 0],
            blackout_points[:, 1],
            blackout_points[:, 2],
            color="red",
            label="Plasma Blackout",
            s=10,
        )

    if estimated_trajectory is not None:
        ax.plot(
            estimated_trajectory[:, 0],
            estimated_trajectory[:, 1],
            estimated_trajectory[:, 2],
            label="Estimated Trajectory",
            color="orange",
            linestyle="--",
            linewidth=2,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("HGV 3D Trajectory Tracking")
    ax.legend()

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

    os.makedirs("assets/results", exist_ok=True)
    plot_trajectory_3d(
        traj["trajectory"], blackout_mask=mask, save_path="assets/results/trajectory_viz.png"
    )
    print("Trajectory plot saved.")
