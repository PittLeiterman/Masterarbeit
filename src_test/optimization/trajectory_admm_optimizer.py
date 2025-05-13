import numpy as np
from scipy.optimize import linprog
from optimization.preparation import evaluate_polynomial_trajectory_time
from optimization.admm import QPProb, admm_solve
import matplotlib.pyplot as plt

def is_point_in_any_convex_region(p, A_list, b_list, tol=1e-5):
    p = np.asarray(p).flatten()
    for A, b in zip(A_list, b_list):
        A = np.asarray(A)
        b = np.asarray(b).flatten()
        if np.all(A @ p <= b + tol):
            return True
    return False

def create_evaluation_matrix(ts, poly_order):
    n_coeff = poly_order + 1
    V = np.vander(ts, n_coeff, increasing=True)  # [len(ts) x n_coeff]
    return V

def build_minimum_snap_qp_with_boundary_conditions(start, goal, poly_order=5):
    n_coeff = poly_order + 1

    Q = np.zeros((n_coeff, n_coeff))
    for i in range(4, n_coeff):
        for j in range(4, n_coeff):
            Q[i, j] = (i * (i - 1) * (i - 2) * (i - 3)) * \
                      (j * (j - 1) * (j - 2) * (j - 3)) / (i + j - 7)

    H = np.block([
        [Q, np.zeros_like(Q)],
        [np.zeros_like(Q), Q]
    ])
    g = np.zeros((2 * n_coeff, 1))

    def basis(t):
        return np.array([t**i for i in range(n_coeff)])

    def deriv_basis(t):
        return np.array([0 if i == 0 else i * t**(i - 1) for i in range(n_coeff)])

    A_eq = np.zeros((8, 2 * n_coeff))
    b_eq = np.zeros((8, 1))

    # Position constraints
    A_eq[0, :n_coeff] = basis(0)
    b_eq[0] = start[0]
    A_eq[1, n_coeff:] = basis(0)
    b_eq[1] = start[1]
    A_eq[2, :n_coeff] = basis(1)
    b_eq[2] = goal[0]
    A_eq[3, n_coeff:] = basis(1)
    b_eq[3] = goal[1]

    # Velocity constraints
    A_eq[4, :n_coeff] = deriv_basis(0)
    A_eq[5, n_coeff:] = deriv_basis(0)
    A_eq[6, :n_coeff] = deriv_basis(1)
    A_eq[7, n_coeff:] = deriv_basis(1)

    return H, g, A_eq, b_eq

def run_admm_trajectory_optimization(initial_points, A_list, b_list, max_iter=20, rho=50.0, alpha=0.98):
    num_eval_points = 60
    poly_order = 5
    n_coeff = poly_order + 1
    ts = np.linspace(0, 1, num_eval_points)

    H_base, g_base, A_eq, b_eq = build_minimum_snap_qp_with_boundary_conditions(
        start=initial_points[0], goal=initial_points[-1], poly_order=poly_order
    )
    n_vars = H_base.shape[0]

    # === Initialisierung ===
    x = np.zeros((n_vars, 1))
    z = np.zeros_like(x)
    lamb = np.zeros((num_eval_points, 2))  # Punkt-Raum
    prev_pts = np.copy(initial_points)

    for k in range(max_iter):
        
        # === lamb → lamb_coeff (via Curve-Fit) ===
        lamb_coeff = np.zeros_like(z)
        for dim in range(2):
            coeffs = np.polyfit(ts, lamb[:, dim], poly_order)[::-1]
            lamb_coeff[dim * n_coeff:(dim + 1) * n_coeff, 0] = coeffs

        # === Primal Step ===
        H = H_base + rho * np.eye(n_vars)
        g = g_base - rho * z + lamb_coeff
        prob = QPProb(H, g, A_eq, b_eq, b_eq)
        x, _, _ = admm_solve(prob, max_iter=50, tol=1e-3)

        # === x → Punkte → Trajektorie ===
        traj_x, traj_y = evaluate_polynomial_trajectory_time(x, num_points=num_eval_points, poly_order=poly_order)
        traj_pts = np.stack([traj_x, traj_y], axis=1)

        # === Projektion / Annäherung ===
        proj_pts = alpha * prev_pts + (1 - alpha) * traj_pts

        # === Abbruchprüfung ===
        for i, pt in enumerate(proj_pts):
            if not is_point_in_any_convex_region(pt, A_list, b_list):
                print(f"[ABBRUCH] Punkt {i} außerhalb bei Iteration {k+1}: {pt}")
                return proj_pts

        # === proj_pts → z_new (Curve-Fit) ===
        z_new = np.zeros_like(z)
        for dim in range(2):
            coeffs = np.polyfit(ts, proj_pts[:, dim], poly_order)[::-1]
            z_new[dim * n_coeff:(dim + 1) * n_coeff, 0] = coeffs

        # === Dual-Update ===
        lamb += rho * (traj_pts - proj_pts)

        # === Update für nächste Runde ===
        z = z_new
        prev_pts = proj_pts

        # === Debug-Plot ===
        plt.figure(figsize=(6, 6))
        plt.plot(initial_points[:, 0], initial_points[:, 1], 'r--', alpha=0.5, label="Initial")
        plt.plot(traj_x, traj_y, 'rx-', markersize=3, label="Primal (Snap)")
        plt.plot(proj_pts[:, 0], proj_pts[:, 1], 'g-', linewidth=2, label="98% Annäherung")
        plt.axis("equal")
        plt.title(f"ADMM Iteration {k + 1}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return proj_pts