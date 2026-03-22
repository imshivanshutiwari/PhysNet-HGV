import numpy as np
from scipy.spatial.distance import cdist


def position_rmse(pred, true):
    pred = np.atleast_2d(pred)
    true = np.atleast_2d(true)
    return np.sqrt(np.mean(np.sum((pred[:, :3] - true[:, :3]) ** 2, axis=1)))


def velocity_rmse(pred, true):
    pred = np.atleast_2d(pred)
    true = np.atleast_2d(true)
    return np.sqrt(np.mean(np.sum((pred[:, 3:6] - true[:, 3:6]) ** 2, axis=1)))


def nees(errors, covariances):
    N, dim = errors.shape
    nees_vals = np.zeros(N)
    for i in range(N):
        err = errors[i]
        cov = covariances[i]
        cov_inv = np.linalg.inv(cov + np.eye(dim) * 1e-6)
        nees_vals[i] = err.T @ cov_inv @ err
    return nees_vals


def track_continuity_pct(estimated_trajectory, true_trajectory, blackout_mask, threshold=500.0):
    if not np.any(blackout_mask):
        return 100.0

    pred_pos = estimated_trajectory[blackout_mask, :3]
    true_pos = true_trajectory[blackout_mask, :3]

    distances = np.linalg.norm(pred_pos - true_pos, axis=1)
    continuous = np.sum(distances < threshold)

    return (continuous / len(distances)) * 100.0


def divergence_rate(estimated_trajectory, true_trajectory, threshold=1000.0):
    pred_pos = estimated_trajectory[:, :3]
    true_pos = true_trajectory[:, :3]
    distances = np.linalg.norm(pred_pos - true_pos, axis=1)

    diverged = np.sum(distances > threshold)
    return diverged / len(distances)


def ospa_distance(x, y, c=100.0, p=2.0):
    m = len(x)
    n = len(y)

    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return c

    x_arr = np.array(x)
    y_arr = np.array(y)

    d = cdist(x_arr, y_arr)
    d = np.minimum(d, c)

    if m == 1 and n == 1:
        return d[0, 0]

    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(d**p)

    cost = d[row_ind, col_ind].sum()
    cost += c**p * abs(m - n)

    return (cost / max(m, n)) ** (1.0 / p)


def gospa_metric(x, y, c=100.0, p=2.0, alpha=2.0):
    return ospa_distance(x, y, c, p)


def pd_pfa_curve(detections, truths, thresholds):
    pd_list = []
    pfa_list = []
    for th in thresholds:
        tp = np.sum((detections >= th) & (truths == 1))
        fp = np.sum((detections >= th) & (truths == 0))
        fn = np.sum((detections < th) & (truths == 1))
        tn = np.sum((detections < th) & (truths == 0))

        pd = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pfa = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        pd_list.append(pd)
        pfa_list.append(pfa)

    return pd_list, pfa_list


def mape(pred, true):
    pred = np.atleast_2d(pred)
    true = np.atleast_2d(true)
    true = np.where(true == 0, 1e-6, true)
    return np.mean(np.abs((true - pred) / true)) * 100


def frechet_distance(P, Q):
    from scipy.spatial.distance import cdist

    dists = cdist(P, Q)
    return np.max(np.min(dists, axis=1))


def reacquisition_time(
    estimated_trajectory, true_trajectory, blackout_mask, recovery_threshold=100.0
):
    recovery_times = {}

    in_blackout = False
    end_idx = -1
    for i, b in enumerate(blackout_mask):
        if b:
            in_blackout = True
        elif in_blackout:
            in_blackout = False
            end_idx = i
            for j in range(end_idx, len(blackout_mask)):
                dist = np.linalg.norm(estimated_trajectory[j, :3] - true_trajectory[j, :3])
                if dist < recovery_threshold:
                    recovery_times[end_idx] = j - end_idx
                    break
            if end_idx not in recovery_times:
                recovery_times[end_idx] = len(blackout_mask) - end_idx

    return recovery_times


def compute_metrics_batch(predictions, targets, blackout_masks):
    pos_rmses = [position_rmse(p, t) for p, t in zip(predictions, targets)]
    vel_rmses = [velocity_rmse(p, t) for p, t in zip(predictions, targets)]

    return {"mean_pos_rmse": np.mean(pos_rmses), "mean_vel_rmse": np.mean(vel_rmses)}
