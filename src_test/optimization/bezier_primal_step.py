import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import comb

# === CONFIG ===
M = 5  # Bézier curve order (quintic)
dim = 2  # 2D trajectories


def bezier_matrix(n, t):
    return np.array([comb(n, i) * (1 - t) ** (n - i) * t ** i for i in range(n + 1)]).T


def snap_cost_matrix(n):
    Q = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            if i >= 4 and j >= 4:
                Q[i, j] = (
                    comb(n, i) * comb(n, j)
                    * math.factorial(i) / math.factorial(i - 4)
                    * math.factorial(j) / math.factorial(j - 4)
                    / (2 * n - i - j + 1)
                )
    return Q

def generate_fixed_bezier_segments(x_opt, n_segments=10, poly_order=5, bezier_order=5):
    n_coeff = poly_order + 1
    n_ctrl = bezier_order + 1
    total_time = len(x_opt) // (2 * n_coeff)

    # Gleichmäßig über die Zeit gesamplete Punkte
    t_vals = np.linspace(0, total_time, n_segments * n_ctrl)
    x_vals, y_vals = [], []

    for t in t_vals:
        seg_idx = min(int(t), total_time - 1)
        local_t = t - seg_idx

        idx_x = (2 * seg_idx) * n_coeff
        idx_y = (2 * seg_idx + 1) * n_coeff

        coeffs_x = x_opt[idx_x:idx_x + n_coeff].flatten()
        coeffs_y = x_opt[idx_y:idx_y + n_coeff].flatten()

        powers = np.array([local_t ** j for j in range(n_coeff)])
        x_vals.append(np.dot(coeffs_x, powers))
        y_vals.append(np.dot(coeffs_y, powers))

    # Punkte in Segmente aufteilen
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    points = np.stack([x_vals, y_vals], axis=1)

    bezier_segments = []
    for i in range(n_segments):
        start = i * n_ctrl
        end = start + n_ctrl
        segment_points = points[start:end]
        ctrl_pts = fit_bezier_from_points(segment_points, bezier_order)
        bezier_segments.append(ctrl_pts)

    return bezier_segments




def reshape_x_opt(x_opt, poly_order=5):
    n_coeff = poly_order + 1
    total_segments = len(x_opt) // (2 * n_coeff)
    x_opt = x_opt.flatten()
    segments = []

    for seg in range(total_segments):
        idx_x = (2 * seg) * n_coeff
        idx_y = (2 * seg + 1) * n_coeff

        coeffs_x = x_opt[idx_x:idx_x + n_coeff]
        coeffs_y = x_opt[idx_y:idx_y + n_coeff]

        segments.append((coeffs_x, coeffs_y))

    return segments


def sample_segment(coeffs, t_vals):
    return np.array([np.polyval(coeffs[::-1], t) for t in t_vals])


def fit_bezier_from_points(points, degree=5):
    t_vals = np.linspace(0, 1, len(points))
    B = np.array([comb(degree, i) * (1 - t_vals) ** (degree - i) * t_vals ** i for i in range(degree + 1)]).T
    ctrl_pts = np.linalg.lstsq(B, points, rcond=None)[0]
    return ctrl_pts


def initialize_beziers_fixed_segments(x_opt, poly_order=5, bezier_order=5, n_segments=10):
    n_coeff = poly_order + 1
    total_segments = len(x_opt) // (2 * (poly_order + 1))
    T = total_segments  # global time range
    ts = np.linspace(0, T, n_segments * (bezier_order + 1))

    t_step = 1 / (bezier_order)

    x_vals, y_vals = [], []
    for t in ts:
        seg_idx = min(int(t), len(x_opt) // (2 * n_coeff) - 1)
        local_t = t - seg_idx

        idx_x = (2 * seg_idx) * n_coeff
        idx_y = (2 * seg_idx + 1) * n_coeff

        coeffs_x = x_opt[idx_x:idx_x + n_coeff].flatten()
        coeffs_y = x_opt[idx_y:idx_y + n_coeff].flatten()

        powers = np.array([local_t ** j for j in range(n_coeff)])
        x_vals.append(np.dot(coeffs_x, powers))
        y_vals.append(np.dot(coeffs_y, powers))

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    bezier_segments = []
    for i in range(n_segments):
        start = i * (bezier_order + 1)
        end = start + (bezier_order + 1)
        points = np.stack([x_vals[start:end], y_vals[start:end]], axis=1)
        ctrl_pts = fit_bezier_from_points(points, bezier_order)
        bezier_segments.append(ctrl_pts)

    return np.array(bezier_segments)


def optimize_primal_step(bezier_segments, Q):
    optimized = []
    N = len(bezier_segments)
    for i, segment in enumerate(bezier_segments):
        X = segment.copy()

        # Fix only the first segment's start and last segment's end
        if i == 0:
            fixed_start = X[0]
        else:
            fixed_start = None

        if i == N - 1:
            fixed_end = X[-1]
        else:
            fixed_end = None

        Q_inner = Q[1:-1, 1:-1]
        rhs_x = np.zeros(M - 1)
        rhs_y = np.zeros(M - 1)

        if fixed_start is not None:
            rhs_x -= Q[1:-1, 0] * fixed_start[0]
            rhs_y -= Q[1:-1, 0] * fixed_start[1]
        if fixed_end is not None:
            rhs_x -= Q[1:-1, -1] * fixed_end[0]
            rhs_y -= Q[1:-1, -1] * fixed_end[1]

        try:
            X_inner_x = np.linalg.solve(Q_inner, rhs_x)
            X_inner_y = np.linalg.solve(Q_inner, rhs_y)
        except np.linalg.LinAlgError:
            X_inner_x = X[1:-1, 0]
            X_inner_y = X[1:-1, 1]

        X[1:-1, 0] = X_inner_x
        X[1:-1, 1] = X_inner_y

        if fixed_start is not None:
            X[0] = fixed_start
        if fixed_end is not None:
            X[-1] = fixed_end

        optimized.append(X)

    return optimized


# === ENTRY POINT FUNCTION ===
def run_primal_step_from_xopt(x_opt, n_segments=10):
    bezier_segments = initialize_beziers_fixed_segments(x_opt, poly_order=5, bezier_order=M, n_segments=n_segments)
    Q = snap_cost_matrix(M)
    optimized_segments = optimize_primal_step(bezier_segments, Q)
    return optimized_segments
