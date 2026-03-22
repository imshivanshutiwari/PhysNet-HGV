import numpy as np
import pytest
from simulation.plasma_model import PlasmaModel
from simulation.hgv_dynamics import HGVDynamics
from simulation.radar_simulator import RadarSimulator
from simulation.trajectory_gen import TrajectoryGenerator


def test_saha_Ne_physical_range():
    pm = PlasmaModel()
    alt = np.array([40000])
    vel = np.array([5000])
    ne = pm.compute_electron_density(alt, vel)
    assert ne.shape == (1,)
    assert ne[0] > 0
    assert ne[0] > 1e10


def test_blackout_triggers_above_threshold():
    pm = PlasmaModel(blackout_threshold=1e18)
    ne = np.array([1e17, 1.5e18, 1e19])
    mask = pm.is_blackout(ne)
    assert np.array_equal(mask, [False, True, True])


def test_6dof_energy_conservation():
    dyn = HGVDynamics()
    initial_state = [0, 0, 50000, 5000, 0, 0]
    traj = dyn.integrate_trajectory(initial_state, duration_s=1.0, controls=[(0, 0, 0)])

    assert len(traj) > 0
    v0 = np.linalg.norm(traj[0, 3:6])
    h0 = np.linalg.norm([traj[0, 0], traj[0, 1], dyn.Re + traj[0, 2]]) - dyn.Re
    E0 = 0.5 * dyn.mass * v0**2 + dyn.mass * dyn.g0 * h0

    v1 = np.linalg.norm(traj[-1, 3:6])
    h1 = np.linalg.norm([traj[-1, 0], traj[-1, 1], dyn.Re + traj[-1, 2]]) - dyn.Re
    E1 = 0.5 * dyn.mass * v1**2 + dyn.mass * dyn.g0 * h1

    assert E1 < E0


def test_radar_returns_correct_shape():
    rs = RadarSimulator()
    true_traj = np.zeros((10, 6))
    mask = np.zeros(10, dtype=bool)
    mask[4:6] = True

    returns = rs.generate_returns(true_traj, mask)
    assert returns.shape == (10, 6)

    assert np.isnan(returns[4]).all()
    assert np.isnan(returns[5]).all()
    assert not np.isnan(returns[3]).any()


def test_trajectory_generator_batch_size():
    gen = TrajectoryGenerator()
    batch = gen.generate_batch(5, parallel=False)
    assert len(batch) == 5
    assert "trajectory" in batch[0]
    assert "radar_returns" in batch[0]
    assert "plasma_profile" in batch[0]
    assert "blackout_mask" in batch[0]
