# core/admm_full.py
from __future__ import annotations
import json, os, time
from math import comb
from typing import List, Tuple, Optional
import numpy as np

# --- your original dependencies (unchanged) ---
from .optimization.primal_step import evaluate_polynomial
from .optimization.minco import precompute_mapping
from .optimization.projection_utils import project_segments_with_coverage
from .utils.cvx_compat import Const
from .utils.bernstein import build_T_block
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

# -------------------------------------------------
# helper: read JSON config
# -------------------------------------------------
def _load_cfg(path: Optional[str]) -> dict:
    defaults = dict(
        rho=1e-6,
        max_iters=200,
        eps=0.25,
        num_segments=30,
        m_per_seg=30,
    )
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # keep only known keys, fall back to defaults
        return {k: data.get(k, v) for k, v in defaults.items()}
    return defaults


# -------------------------------------------------
# tiny helpers from your function (unchanged logic)
# -------------------------------------------------
def straight_line_path(start_xy, goal_xy, num_nodes):
    xs = np.linspace(start_xy[0], goal_xy[0], num_nodes)
    ys = np.linspace(start_xy[1], goal_xy[1], num_nodes)
    return np.column_stack([xs, ys])

def bernstein_basis_row(n, u):
    um = 1.0 - u
    return np.array([comb(n, k) * (um**(n-k)) * (u**k) for k in range(n+1)], dtype=float)

def bezier_curve_from_cpoints(C_seg, res=800):
    C = np.asarray(C_seg, float).reshape(-1, 2)
    n = C.shape[0] - 1
    u = np.linspace(0.0, 1.0, res)
    pts = np.empty((res, 2))
    for i, ui in enumerate(u):
        B = bernstein_basis_row(n, ui)
        pts[i] = B @ C
    return pts[:,0], pts[:,1]

def admm_residuals_cp(Acx, Acy, xi_x, xi_y, Z, Z_prev, rho_val, bcx, bcy):
    xi_x = np.asarray(xi_x).ravel()
    xi_y = np.asarray(xi_y).ravel()
    rx = Acx @ xi_x + bcx - Z[:, 0]
    ry = Acy @ xi_y + bcy - Z[:, 1]
    r_inf = max(np.linalg.norm(rx, np.inf), np.linalg.norm(ry, np.inf))
    dZ = Z - Z_prev
    sx = rho_val * (Acx.T @ dZ[:, 0])
    sy = rho_val * (Acy.T @ dZ[:, 1])
    s_inf = max(np.linalg.norm(sx, np.inf), np.linalg.norm(sy, np.inf))
    return r_inf, s_inf

def update_rho_osqp_with_s_cp(rho, r_inf, s_inf, rho_min=1e-6, rho_max=1e6, step_limit=5.0, eps=1e-12):
    if r_inf < eps and s_inf < eps:
        return rho
    scale = np.sqrt(r_inf / max(s_inf, eps))
    scale = float(np.clip(scale, 1.0/step_limit, step_limit))
    return float(np.clip(rho * scale, rho_min, rho_max))

def _cell_center(rc: Tuple[int,int]) -> Tuple[float,float]:
    r, c = rc
    return (c + 0.5, r + 0.5)

# -------------------------------------------------
# PUBLIC: run your ADMM on the app’s decomposition
# -------------------------------------------------
def run_admm_from_app(
    A_list, b_list,
    simplified_cells: List[Tuple[int,int]],
    start_cell: Tuple[int,int],
    goal_cell: Tuple[int,int],
    grid_cols: int,
    grid_rows: int,
    config_path: Optional[str] = None,
    capture_iterations: bool = False,         # NEW
    capture_every: int = 1,                   # NEW
    iter_plot_samples: int = 12,              # NEW (per segment, per iteration)
):
    """
    Returns a dict with:
      - 'segments_samples': [ (m_i,2) arrays ] sampled trajectory per segment (grid units)
      - 'ctrl_points':      (6*S, 2) final projected control points Z (stacked)
      - 'assign':           list[int] region index per segment
      - plus a few sizes (S, ctrl_per_seg)
    """
    cfg = _load_cfg(config_path)

    # --- Build environment directly from the app ---
    # area_size from the grid, start/goal from cell centers
    area_size = (grid_cols, grid_rows)
    start_xy = _cell_center(start_cell)
    goal_xy  = _cell_center(goal_cell)

    # simplified polyline in (x,y) cell-centers
    if not simplified_cells or len(simplified_cells) < 2:
        raise RuntimeError("Simplified path is empty – run A* and simplification first.")

    path_real = np.array([_cell_center(rc) for rc in simplified_cells], dtype=float)  # (N,2)

    # --- Initial segmentization straight from start->goal (exactly like your code) ---
    S = int(cfg.get("num_segments") or (len(path_real) - 1))
    if S < 1:
        S = 1
    init_path = straight_line_path(start_xy, goal_xy, S + 1)
    p_all_x = init_path[:, 0]
    p_all_y = init_path[:, 1]

    def allocate_times_from_chords(path_xy, v_des: float = 1.0, t_min: float = 0.05):
        p = np.asarray(path_xy, float)
        if p.shape[0] < 2:
            return np.array([0.0, 1.0])
        chords = np.linalg.norm(np.diff(p, axis=0), axis=1)
        v_des = max(float(v_des), 1e-6)
        T_i = np.maximum(chords / v_des, float(t_min))
        return np.concatenate(([0.0], np.cumsum(T_i)))

    segment_times = allocate_times_from_chords(init_path)  # uses defaults 1.0, 0.05
    S = len(segment_times) - 1


    # --- Minimum-snap mapping a = M xi + c (per axis) ---
    v_start = tuple(cfg.get("v_start", (0.0, 0.0)))
    v_end   = tuple(cfg.get("v_end",   (0.0, 0.0)))

    coeffs_from_xi_x, Mx, cx, Q_blk = precompute_mapping(
        segment_times, p0=p_all_x[0], pS=p_all_x[-1],
        v_start=0.0, v_end=0.0
    )
    coeffs_from_xi_y, My, cy, _ = precompute_mapping(
        segment_times, p0=p_all_y[0], pS=p_all_y[-1],
        v_start=0.0, v_end=0.0
    )

    # interior decision variables (initial guess = interior waypoints)
    xi_x = p_all_x[1:-1].copy()
    xi_y = p_all_y[1:-1].copy()

    # sampling operator Φ for control points in Bernstein basis (same as before)
    T_blk = build_T_block(segment_times, degree=5)

    # reduced snap terms
    Hx = (Mx.T @ (Q_blk @ Mx))
    fx = (Mx.T @ (Q_blk @ cx))
    Hy = (My.T @ (Q_blk @ My))
    fy = (My.T @ (Q_blk @ cy))

    Acx = T_blk @ Mx;  bcx = T_blk @ cx
    Acy = T_blk @ My;  bcy = T_blk @ cy

    ctrl_per_seg   = T_blk.shape[0] // S       # 6 for quintic
    coeffs_per_seg = Mx.shape[0] // S          # 6 for quintic

    # initial coefficients from xi
    a_x_stacked = coeffs_from_xi_x(xi_x)       # (6S,)
    a_y_stacked = coeffs_from_xi_y(xi_y)

    # control points X from a (power -> Bernstein via T_blk)
    Cx0 = T_blk @ a_x_stacked
    Cy0 = T_blk @ a_y_stacked
    X = np.column_stack([Cx0, Cy0])            # ((6S) x 2)

    # project each segment to best convex region from the app’s decomposition
    C_segments = [X[ctrl_per_seg*i : ctrl_per_seg*(i+1), :] for i in range(S)]
    z_traj, assign, _ = project_segments_with_coverage(C_segments, A_list, b_list)
    u_traj = np.zeros_like(z_traj)

    # ADMM loop (unchanged)
    rho = float(cfg.get("rho", 1.0))
    rho_cache = None
    z_traj_prev = z_traj.copy()
    m_per_seg = int(cfg.get("m_per_seg", 10))

    iter_history = []

    for k in range(int(cfg.get("max_iters", 80))):
        # keep previous Z before projection for correct residuals
        Z_prev_for_res = z_traj.copy()        # NEW

        ZU = z_traj - u_traj

        # cache factorization when rho changes
        if (k == 0) or (rho_cache is None) or (abs(rho_cache - rho) > 0):
            LHSx = Hx + rho * (Acx.T @ Acx)
            LHSy = Hy + rho * (Acy.T @ Acy)
            Lx_factor = splu(csc_matrix(LHSx))
            Ly_factor = splu(csc_matrix(LHSy))
            rho_cache = rho

        RHSx = rho * (Acx.T @ (ZU[:, 0] - bcx)) - fx
        RHSy = rho * (Acy.T @ (ZU[:, 1] - bcy)) - fy

        xi_x = Lx_factor.solve(RHSx)
        xi_y = Ly_factor.solve(RHSy)

        # recover coefficients
        a_x_stacked = coeffs_from_xi_x(xi_x)
        a_y_stacked = coeffs_from_xi_y(xi_y)

        coeffs_x = [Const(a_x_stacked[coeffs_per_seg*i : coeffs_per_seg*(i+1)]) for i in range(S)]
        coeffs_y = [Const(a_y_stacked[coeffs_per_seg*i : coeffs_per_seg*(i+1)]) for i in range(S)]

        # current control points X (from a)
        Cx = T_blk @ a_x_stacked
        Cy = T_blk @ a_y_stacked
        X  = np.column_stack([Cx, Cy])          # ((6S) x 2)

        # projection (per segment) -> new z_traj & assignment
        C_segments = [X[ctrl_per_seg*i : ctrl_per_seg*(i+1), :] for i in range(S)]
        z_traj, assign, _ = project_segments_with_coverage(C_segments, A_list, b_list)

        # residuals: use previous Z (Z_prev_for_res) vs new Z (z_traj)
        r_inf, s_inf = admm_residuals_cp(Acx, Acy, xi_x, xi_y, z_traj, Z_prev_for_res, rho, bcx, bcy)

        # dual update
        u_traj = u_traj + (X - z_traj)

        # adaptive rho (unchanged)
        if k >= 3 and (k % 5 == 0):
            rho_new = update_rho_osqp_with_s_cp(rho, r_inf, s_inf, rho_min=1e-6, rho_max=1e6, step_limit=5.0)
            if rho_new != rho:
                scale = rho / rho_new
                u_traj = scale * u_traj
                rho = rho_new
                rho_cache = None

        # --- SNAPSHOT (optional) ---
        if capture_iterations and (k % max(1,int(capture_every)) == 0):
            # light-weight samples of the current *curve* for visualization
            seg_samples_k = []
            for i in range(S):
                dt = segment_times[i+1] - segment_times[i]
                t_vals = np.linspace(0, dt, int(iter_plot_samples))
                ax_i = coeffs_x[i].value
                ay_i = coeffs_y[i].value
                xs = [evaluate_polynomial(ax_i, t) for t in t_vals]
                ys = [evaluate_polynomial(ay_i, t) for t in t_vals]
                seg_samples_k.append(np.column_stack((xs, ys)))
            iter_history.append({
                "k": int(k),
                "rho": float(rho),
                "r_inf": float(r_inf),
                "s_inf": float(s_inf),
                "max_diff": float(np.max(np.abs(X - z_traj))),
                "segments_samples": seg_samples_k,  # list of (m,2)
                "z_ctrl": z_traj.copy(),            # ((6S),2) projected control points
                "assign": np.array(assign, dtype=int)  # per-seg region id
            })

        # convergence
        max_diff = float(np.max(np.abs(X - z_traj)))
        if max_diff < float(cfg.get("eps", 1e-3)):
            break

    # final sampling (as before) ...
    segments_samples = []
    for i in range(S):
        dt = segment_times[i+1] - segment_times[i]
        t_vals = np.linspace(0, dt, m_per_seg)
        a_x_seg = coeffs_x[i].value
        a_y_seg = coeffs_y[i].value
        x_vals = [evaluate_polynomial(a_x_seg, t) for t in t_vals]
        y_vals = [evaluate_polynomial(a_y_seg, t) for t in t_vals]
        segments_samples.append(np.column_stack((x_vals, y_vals)))

    return {
        "segments_samples": segments_samples,
        "ctrl_points": z_traj,
        "assign": assign,
        "S": S,
        "ctrl_per_seg": ctrl_per_seg,
        "iterations": iter_history,     # NEW
    }
