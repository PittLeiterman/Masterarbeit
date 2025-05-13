import numpy as np
from scipy.special import factorial
import cvxpy as cp
from optimization.admm import QPProb, admm_solve
from matplotlib import pyplot as plt
from optimization.preparation import evaluate_polynomial_trajectory_time

def project_point_onto_convex_set(p, A, b):
    """
    Projektion eines Punkts p = [x, y] auf die konvexe Menge {x | Ax ≤ b}
    Rückgabe: projizierter Punkt (2D)
    """
    x_var = cp.Variable(2)
    objective = cp.Minimize(cp.sum_squares(x_var - p))
    b = np.asarray(b).flatten()
    constraints = [A @ x_var <= b]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)
    return x_var.value

def project_onto_convex_regions(x, A_list, b_list, segment_times, poly_order=7):
    """
    Projiziert die Endpunkte jedes Segments auf die zugehörigen konvexen Regionen.
    
    Args:
        x: Optimierungsvariable x (Koeffizientenvektor, shape (n,))
        A_list, b_list: Liste der Constraints pro Segment (A @ x ≤ b)
        segment_times: Liste der Segmentlängen
        poly_order: Grad der Polynome (Standard: 7 → 8 Koeffizienten)
    
    Returns:
        z: projizierter Punkt-Vektor (gleiche Dimension wie x)
    """
    n_segments = len(segment_times)
    n_coeffs = poly_order + 1
    z = x.copy()

    for i in range(n_segments):
        T = segment_times[i]
        idx_x = (i * 2 + 0) * n_coeffs
        idx_y = (i * 2 + 1) * n_coeffs
        coeffs_x = x[idx_x:idx_x + n_coeffs].flatten()
        coeffs_y = x[idx_y:idx_y + n_coeffs].flatten()

        # Evaluiere Endpunkt bei T
        powers = np.array([T**j for j in range(n_coeffs)])
        end_x = np.dot(coeffs_x, powers)
        end_y = np.dot(coeffs_y, powers)
        p = np.array([end_x, end_y])

        # Projiziere auf die passende Region
        A = A_list[i]
        b = b_list[i]
        proj = project_point_onto_convex_set(p, A, b)

        # Setze neuen Endpunkt, indem wir a₀ anpassen (für Start bei p_proj)
        z[idx_x + 0] += proj[0] - end_x  # Verschiebe a₀ in x
        z[idx_y + 0] += proj[1] - end_y  # Verschiebe a₀ in y

    return z

def build_minimum_snap_qp_primal(num_segments, segment_times, poly_order=7):
    """
    Erzeugt H, g für Snap-Minimierung bei festen Zeiten und num_segments.
    """
    dim = 2  # x und y
    n_coeffs = poly_order + 1
    total_vars = num_segments * n_coeffs * dim
    H = np.zeros((total_vars, total_vars))
    g = np.zeros((total_vars, 1))

    for seg in range(num_segments):
        T = segment_times[seg]
        for d in range(dim):
            offset = (seg * 2 + d) * n_coeffs
            H_seg = np.zeros((n_coeffs, n_coeffs))
            for i in range(4, n_coeffs):
                for j in range(4, n_coeffs):
                    H_seg[i, j] = (
                        factorial(i) / factorial(i - 4)
                        * factorial(j) / factorial(j - 4)
                        / (i + j - 7)
                        * T ** (i + j - 7)
                    )
            H[offset:offset+n_coeffs, offset:offset+n_coeffs] = H_seg
    return H, g


def build_initial_derivative_constraints_primal(num_segments, start, velocity=None, poly_order=7):
    """
    Erzeugt Startconstraints für Position (und optional Geschwindigkeit)
    """
    dim = 2
    n_coeffs = poly_order + 1
    total_vars = num_segments * n_coeffs * dim
    rows = []
    l = []
    u = []

    # Position
    for d in range(dim):
        A_row = np.zeros((1, total_vars))
        A_row[0, d * n_coeffs + 0] = 1  # t^0
        rows.append(A_row)
        l.append([start[d]])
        u.append([start[d]])

    # Geschwindigkeit (optional)
    if velocity is not None:
        for d in range(dim):
            A_row = np.zeros((1, total_vars))
            A_row[0, d * n_coeffs + 1] = 1  # Ableitung t^1
            rows.append(A_row)
            l.append([velocity[d]])
            u.append([velocity[d]])

    A = np.vstack(rows)
    l = np.vstack(l)
    u = np.vstack(u)
    return A, l, u


def build_terminal_derivative_constraints_primal(num_segments, goal, velocity=None, segment_times=None, poly_order=7):
    """
    Erzeugt Endconstraints für Position (und optional Geschwindigkeit) für das letzte Segment.
    """
    dim = 2
    n_coeffs = poly_order + 1
    total_vars = num_segments * n_coeffs * dim
    rows = []
    l = []
    u = []
    T = segment_times[-1]

    for d in range(dim):
        A_row = np.zeros((1, total_vars))
        for i in range(n_coeffs):
            index = ((num_segments - 1) * 2 + d) * n_coeffs + i
            A_row[0, index] = T ** i
        rows.append(A_row)
        l.append([goal[d]])
        u.append([goal[d]])

    # Geschwindigkeit (optional)
    if velocity is not None:
        for d in range(dim):
            A_row = np.zeros((1, total_vars))
            for i in range(1, n_coeffs):
                index = ((num_segments - 1) * 2 + d) * n_coeffs + i
                A_row[0, index] = i * T ** (i - 1)
            rows.append(A_row)
            l.append([velocity[d]])
            u.append([velocity[d]])

    A = np.vstack(rows)
    l = np.vstack(l)
    u = np.vstack(u)
    return A, l, u



def build_continuity_constraints_primal(num_segments, segment_times, order=3, poly_order=7):
    """
    Erzwingt C^order-Kontinuität an Segmentübergängen.
    """
    dim = 2
    n_coeffs = poly_order + 1
    total_vars = num_segments * n_coeffs * dim
    rows = []
    l = []
    u = []

    for seg in range(num_segments - 1):
        T = segment_times[seg]
        for d in range(dim):
            for k in range(order + 1):  # Ableitungsgrad
                A_row = np.zeros((1, total_vars))

                # rechter Rand Segment seg
                for i in range(k, n_coeffs):
                    A_row[0, (seg * dim + d) * n_coeffs + i] = (
                        factorial(i) / factorial(i - k) * T ** (i - k)
                    )

                # linker Rand Segment seg+1
                for i in range(k, n_coeffs):
                    A_row[0, ((seg + 1) * dim + d) * n_coeffs + i] -= (
                        factorial(i) / factorial(i - k) * 0 ** (i - k)
                    )

                rows.append(A_row)
                l.append([0.0])
                u.append([0.0])

    A = np.vstack(rows)
    l = np.vstack(l)
    u = np.vstack(u)
    return A, l, u

import cvxpy as cp

def solve_qp_cvxpy(H, g, A, l, u):
    x = cp.Variable(H.shape[0])
    objective = cp.Minimize(0.5 * cp.quad_form(x, H) + g.T @ x)
    constraints = [A @ x >= l.flatten(), A @ x <= u.flatten()]
    prob = cp.Problem(objective, constraints)
    print("Lösen des QP-Problems...")
    prob.solve(solver=cp.OSQP)  # alternativ: SCS, ECOS, etc.
    return x.value.reshape((H.shape[0], 1))



def plot_segment_lengths(x_opt, segment_times, poly_order=7, num_points=100, ax=None):
    """
    Zeichnet die räumlichen Trajektorien jedes Segments (z.B. aus dem Primal Step)
    in ein gegebenes matplotlib-Axes-Objekt.

    Parameters:
    - x_opt: Optimierte Polynomkoeffizienten
    - segment_times: Liste der Zeitdauern je Segment
    - poly_order: Grad des Polynoms (Standard: 7 → 8 Koeffizienten)
    - num_points: Auflösung pro Segment
    - ax: matplotlib Axes-Objekt (z. B. aus Hauptplot). Falls None → aktuelles wird verwendet.
    
    Returns:
    - lengths: Liste der räumlichen Längen je Segment
    """
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    num_segments = len(segment_times)
    n_coeffs = poly_order + 1
    lengths = []

    for i in range(num_segments):
        coeffs_x = x_opt[i * 2 * n_coeffs:(i * 2 + 1) * n_coeffs].flatten()
        coeffs_y = x_opt[(i * 2 + 1) * n_coeffs:(i * 2 + 2) * n_coeffs].flatten()

        ts = np.linspace(0, segment_times[i], num_points)
        xs = np.polyval(coeffs_x[::-1], ts)
        ys = np.polyval(coeffs_y[::-1], ts)

        dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        lengths.append(np.sum(dists))

        # Segmentpfad in Plot einzeichnen
        ax.plot(xs, ys, linestyle='dotted', alpha=0.7, label=f"Segment {i} ({lengths[-1]:.2f} m)")

    return lengths


def solve_primal_step_with_admm(num_segments, segment_times, start, goal, velocity_start=None, velocity_end=None, z=None, lamb=None, rho=1.0):
    H, g = build_minimum_snap_qp_primal(num_segments, segment_times)

    # Augmented terms:
    if z is not None and lamb is not None:
        g = g - rho * (z - lamb / rho)  # Equivalent to shifting linear term
        H = H + rho * np.eye(H.shape[0])  # Augment diagonals

    A_init, l_init, u_init = build_initial_derivative_constraints_primal(
        num_segments, start=start, velocity=velocity_start
    )
    A_term, l_term, u_term = build_terminal_derivative_constraints_primal(
        num_segments, goal=goal, velocity=velocity_end, segment_times=segment_times
    )
    A_cont, l_cont, u_cont = build_continuity_constraints_primal(
        num_segments, segment_times, order=3
    )

    A = np.vstack([A_init, A_term, A_cont])
    l = np.vstack([l_init, l_term, l_cont])
    u = np.vstack([u_init, u_term, u_cont])

    prob = QPProb(H, g, A, l, u)
    x_opt, _, _ = admm_solve(prob, tol=1e-3, max_iter=1000)
    return x_opt


def admm_trajectory_opt(
    num_segments,
    segment_times,
    start,
    goal,
    velocity_start,
    velocity_end,
    convex_regions,
    max_iter=50,
    tol=1e-3,
    rho=1.0,
    debug=False
):
    n_vars = num_segments * (7 + 1) * 2  # poly_order + 1, dim=2
    x = np.zeros((n_vars, 1))
    z = np.zeros_like(x)
    lamb = np.zeros_like(x)

    for i in range(max_iter):
        # x-Update: Primal Step mit Augmented Termen
        x = solve_primal_step_with_admm(
            num_segments,
            segment_times,
            start=start,
            goal=goal,
            velocity_start=velocity_start,
            velocity_end=velocity_end,
            z=z,
            lamb=lamb,
            rho=rho
        )

        # z-Update: Projektion auf konvexe Regionen
        z_old = z.copy()
        A_list, b_list = convex_regions
        z = project_onto_convex_regions(x + lamb / rho, A_list, b_list, segment_times)

        # λ-Update
        lamb += rho * (x - z)

        # Residuen
        primal_res = np.linalg.norm(x - z)
        dual_res = np.linalg.norm(z - z_old)

        if debug:
            print(f"Iter {i:3d}: r_p = {primal_res:.2e}, r_d = {dual_res:.2e}")

            # Debug-Plot der aktuellen x- und z-Trajektorie
            traj_x_x, traj_y_x = evaluate_polynomial_trajectory_time(x, segment_times, poly_order=7)
            traj_x_z, traj_y_z = evaluate_polynomial_trajectory_time(z, segment_times, poly_order=7)

            plt.figure(figsize=(6, 6))
            plt.plot(traj_x_x, traj_y_x, label="x (Primal Step)", color="purple", linewidth=2)
            plt.plot(traj_x_z, traj_y_z, label="z (Projektion)", color="green", linestyle="--")
            plt.scatter(traj_x_x[-1], traj_y_x[-1], c="purple", marker="x", label="Endpunkt x")
            plt.scatter(traj_x_z[-1], traj_y_z[-1], c="green", marker="o", label="Endpunkt z")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.title(f"ADMM Iteration {i+1}")
            plt.pause(0.5)  # Optional: kurze Pause zum Anschauen
            plt.close()


        if primal_res < tol and dual_res < tol:
            print(f"Converged in {i + 1} iterations.")
            break

    return x
