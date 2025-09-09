def run_admm_trajectory_optimization(config, DEBUG=False):    
    from input.trees2D import create_occupancy_grid, load_forest_from_file
    from pathfinder.AStar import astar
    from utils.path_manipulation import simplify_path

    from optimization.primal_step import evaluate_polynomial
    from optimization.projection_utils import project_segments_to_convex_regions

    from optimization.minco import precompute_mapping
    from optimization.sampling import build_Phi
    from utils.cvx_compat import Const
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu


    import pydecomp as pdc
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    import numpy as np
    import time

    
    def straight_line_path(start_xy, goal_xy, num_nodes):
        xs = np.linspace(start_xy[0], goal_xy[0], num_nodes)
        ys = np.linspace(start_xy[1], goal_xy[1], num_nodes)
        return np.column_stack([xs, ys])  # (num_nodes, 2)

    


    def admm_residuals_stacked(Phi, a_x_vec, a_y_vec, z_traj, z_traj_prev, rho_val):
        Z  = np.vstack(z_traj)
        Zp = np.vstack(z_traj_prev)

        # ensure 1D
        a_x_vec = np.asarray(a_x_vec).ravel()
        a_y_vec = np.asarray(a_y_vec).ravel()

        rx = Phi @ a_x_vec - Z[:, 0]
        ry = Phi @ a_y_vec - Z[:, 1]
        r_inf = max(np.linalg.norm(rx, np.inf), np.linalg.norm(ry, np.inf))

        dz = Z - Zp
        sx = rho_val * (Phi.T @ dz[:, 0])
        sy = rho_val * (Phi.T @ dz[:, 1])
        s_inf = max(np.linalg.norm(sx, np.inf), np.linalg.norm(sy, np.inf))

        return r_inf, s_inf





    def update_rho_osqp_with_s(
            rho, a_x_stacked, a_y_stacked, Phi, Z, U, Q_blk, s_inf,
            rho_min=1e-6, rho_max=1e6, step_limit=5.0, eps=1e-12):
        # --- primal residual norm (∞) ---
        Ax_x = Phi @ a_x_stacked
        Ax_y = Phi @ a_y_stacked
        z_x, z_y = Z[:, 0], Z[:, 1]
        r_p_inf = max(np.linalg.norm(Ax_x - z_x, np.inf),
                    np.linalg.norm(Ax_y - z_y, np.inf))
        denom_p = max(np.linalg.norm(Ax_x, np.inf),
                    np.linalg.norm(Ax_y, np.inf),
                    np.linalg.norm(z_x,  np.inf),
                    np.linalg.norm(z_y,  np.inf), 1.0)
        r_scaled = r_p_inf / max(denom_p, eps)

        # --- dual side ---
        Px_x = Q_blk @ a_x_stacked
        Px_y = Q_blk @ a_y_stacked
        Aty_x = rho * (Phi.T @ U[:, 0])
        Aty_y = rho * (Phi.T @ U[:, 1])
        denom_d = max(np.linalg.norm(Px_x, np.inf),
                    np.linalg.norm(Px_y, np.inf),
                    np.linalg.norm(Aty_x, np.inf),
                    np.linalg.norm(Aty_y, np.inf), 1.0)
        s_scaled = s_inf / max(denom_d, eps)

        if r_scaled < eps and s_scaled < eps:
            return rho

        scale = np.sqrt(r_scaled / max(s_scaled, eps))
        # clamp update (avoid too large jumps)
        scale = float(np.clip(scale, 1.0/step_limit, step_limit))
        rho_new = float(np.clip(rho * scale, rho_min, rho_max))
        return rho_new





    # Parameter
    area_size = tuple(config["area_size"])
    shape = config["shape"]
    start = tuple(config["start"])
    goal = tuple(config["goal"])
    keep = config["keep"]
    rho = config["rho"]
    max_iters = config["max_iters"]
    eps = config["eps"]
    v_start = tuple(config["v_start"])
    v_end = tuple(config["v_end"])
    num_segments = config.get("num_segments")
    m_per_seg = int(config.get("m_per_seg", 10))


    # Baum- und Grid-Erzeugung
    obstacles = load_forest_from_file(f"input/data/{shape}.txt")
    grid = create_occupancy_grid(obstacles, area_size, area_size, tree_size=1)

    print("Hindernisse generiert")

    # Umgedreht (x, y) für pydecomp & plotting
    start_xy = (start[1], start[0])
    goal_xy = (goal[1], goal[0])


    # A* Pfadsuche
    path = astar(grid, start, goal)

    # Pfad prüfen
    if path is None or len(path) == 0:
        print("Kein Pfad gefunden!")
        # Plot Setup
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_title("Hindernisse (Debug-Ansicht ohne Pfad)")
        ax.set_xlim(0, area_size[0])
        ax.set_ylim(0, area_size[1])
        ax.grid(True)

        # Hindernisse
        forest_np = np.array(obstacles)
        ax.plot(forest_np[:, 0], forest_np[:, 1], 'go', markersize=3, label='Bäume')

        # Start & Ziel
        ax.plot(start_xy[0], start_xy[1], 'bo', markersize=6, label='Start')
        ax.plot(goal_xy[0], goal_xy[1], 'ro', markersize=6, label='Ziel')

        ax.legend()
        plt.show()
        exit()

    print(f"Pfad gefunden")
    # Pfad Vereinfachung
    path_simplified = simplify_path(path, keep)

    # Für pydecomp
    path_real = np.array([(y, x) for (x, y) in path_simplified])

    if DEBUG:
        # Plot Setup
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_title("Pfad & Hindernisse (Debug-Ansicht)")
        ax.set_xlim(0, area_size[0])
        ax.set_ylim(0, area_size[1])
        ax.grid(True)

        # Hindernisse
        forest_np = np.array(obstacles)
        ax.plot(forest_np[:, 0], forest_np[:, 1], 'go', markersize=3, label='Bäume')

        # Start & Ziel
        ax.plot(start_xy[0], start_xy[1], 'bo', markersize=6, label='Start')
        ax.plot(goal_xy[0], goal_xy[1], 'ro', markersize=6, label='Ziel')

        # Originalpfad (von A*)
        if path:
            path_np = np.array([(y, x) for (x, y) in path])  # (x, y)
            ax.plot(path_np[:, 0], path_np[:, 1], 'k--', label='A*-Pfad')

        # Vereinfachter Pfad
        if path_simplified:
            simplified_np = np.array([(y, x) for (x, y) in path_simplified])
            ax.plot(simplified_np[:, 0], simplified_np[:, 1], 'b-', linewidth=2, label='Vereinfachter Pfad')

        ax.legend()
        plt.show()
        exit()


    # Konvexe Zerlegung
    A_list, b_list = pdc.convex_decomposition_2D(obstacles, path_real, np.array([area_size]))

    print("Konvexe Zerlegung abgeschlossen")

    ax = pdc.visualize_environment(Al=A_list, bl=b_list, p=path_real, planar=True)

    ax.plot(start_xy[0], start_xy[1], 'go', label='Start')
    ax.plot(goal_xy[0], goal_xy[1], 'bo', label='Goal')

    ax.plot(obstacles[:, 0], obstacles[:, 1], "o", color="green", label="Bäume")

    # -------- (1) Knot positions: straight line from start -> goal --------
    S = int(num_segments)
    init_path = straight_line_path(start_xy, goal_xy, S + 1)  # (S+1, 2)
    p_all_x = init_path[:, 0]
    p_all_y = init_path[:, 1]

    # -------- (2) Segment times: equal/chord-based on the straight line --------
    def allocate_times_from_chords(path_xy, v_des=1.0, t_min=0.05):
        p = np.asarray(path_xy, float)
        chords = np.linalg.norm(np.diff(p, axis=0), axis=1)   # (S,)
        v_des = max(float(v_des), 1e-6)
        T_i = np.maximum(chords / v_des, float(t_min))        # per-segment durations
        return np.concatenate(([0.0], np.cumsum(T_i)))        # (S+1,)

    segment_times = allocate_times_from_chords(
        init_path,
        v_des=float(config.get("v_des", 1.0)),
        t_min=float(config.get("t_min", 0.05))
    )
    S = len(segment_times) - 1


    # Build minimum-snap mapping a(xi) = M xi + c for each axis
    coeffs_from_xi_x, Mx, cx, Q_blk = precompute_mapping(
        segment_times,
        p0=p_all_x[0], pS=p_all_x[-1],
        v_start=v_start[0], v_end=v_end[0]
    )
    coeffs_from_xi_y, My, cy, _ = precompute_mapping(
        segment_times,
        p0=p_all_y[0], pS=p_all_y[-1],
        v_start=v_start[1], v_end=v_end[1]
    )

    # Interior decision variables (initial guess = interior waypoints)
    xi_x = p_all_x[1:-1].copy()
    xi_y = p_all_y[1:-1].copy()

    # Build sampling operator Φ with m_per_seg samples per segment
    Phi = build_Phi(segment_times, m_per_seg)   # shape ((S*m) x 6S)

    # Reduced snap terms: H = M^T (2Q) M,  f = M^T (2Q) c
    Hx = (Mx.T @ (Q_blk @ Mx))
    fx = (Mx.T @ (Q_blk @ cx))
    Hy = (My.T @ (Q_blk @ My))
    fy = (My.T @ (Q_blk @ cy))

    # Reduced sampling: A = Φ M ,  b = Φ c
    Ax = Phi @ Mx
    bx = Phi @ cx
    Ay = Phi @ My
    by = Phi @ cy

    # Helper: build initial coefficients from xi for plotting/projection
    def current_coeffs_from_xi(xi_x, xi_y):
        a_x = coeffs_from_xi_x(xi_x)
        a_y = coeffs_from_xi_y(xi_y)
        coeffs_x = [Const(a_x[6*i:6*(i+1)]) for i in range(S)]
        coeffs_y = [Const(a_y[6*i:6*(i+1)]) for i in range(S)]
        return coeffs_x, coeffs_y

    # Build initial coefficients (instead of CVXPY/solve_axis)
    coeffs_x, coeffs_y = current_coeffs_from_xi(xi_x, xi_y)


    projected_x, projected_y, reassigned = project_segments_to_convex_regions(
            coeffs_x, coeffs_y, segment_times, A_list, b_list, n_samples=m_per_seg
        )

    colors = cm.viridis(np.linspace(0, 1, len(segment_times) - 1))

    segment_lengths = []

    for i in range(len(segment_times) - 1):
        t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], m_per_seg)
        a_x = coeffs_x[i].value
        a_y = coeffs_y[i].value

        x_vals = [evaluate_polynomial(a_x, t) for t in t_vals]
        y_vals = [evaluate_polynomial(a_y, t) for t in t_vals]
        
        ax.plot(x_vals, y_vals, color=colors[i], label=f'Segment {i+1}')
        dist = sum(np.hypot(np.diff(x_vals), np.diff(y_vals)))
        segment_lengths.append(dist)

    # First draw all in red
    for x_vals, y_vals in zip(projected_x, projected_y):
        ax.plot(x_vals, y_vals, 'r-', linewidth=2, 
                label="projiziert" if "projiziert" not in ax.get_legend_handles_labels()[1] else "")

    # Then overwrite reassigned ones in magenta
    for idx, (x_vals, y_vals) in enumerate(zip(projected_x, projected_y)):
        if any(r[0] == idx for r in reassigned):
            ax.plot(
                x_vals, y_vals,
                color="magenta", linewidth=4, linestyle="--", marker="o", markersize=8,
                label="reassigned" if "reassigned" not in ax.get_legend_handles_labels()[1] else ""
            )

    plt.show()


    # Dummy-Initialisierung für z und u
    z_traj = [np.column_stack((x, y)) for x, y in zip(projected_x, projected_y)]

    # Erste Iteration: berechne u^1
    x_traj = []
    for i in range(len(segment_times) - 1):
        t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], m_per_seg)
        a_x = coeffs_x[i].value
        a_y = coeffs_y[i].value
        x_vals = [evaluate_polynomial(a_x, t) for t in t_vals]
        y_vals = [evaluate_polynomial(a_y, t) for t in t_vals]
        x_traj.append(np.column_stack((x_vals, y_vals)))

    u_traj = [x - z for x, z in zip(x_traj, z_traj)]  # u^1

    rho_list = []

    fig_rho, ax_rho   = plt.subplots(figsize=(6, 3))

    for k in range(max_iters):
        print(f"--- Iteration {k+1} ---")
        start_iter = time.perf_counter()

        # --- Build ZU (same as before) ---
        if k == 0:
            z_traj_prev = [z.copy() for z in z_traj]

        Z  = np.vstack(z_traj)    # ((S*m) x 2)
        U  = np.vstack(u_traj)    # ((S*m) x 2)
        ZU = Z - U

        # --- Cache factorization when rho is unchanged ---
        if k == 0:
            rho_cache = None
        if (k == 0) or (rho_cache is None) or (abs(rho_cache - rho) > 0):
            LHSx = Hx + rho * (Ax.T @ Ax)
            LHSy = Hy + rho * (Ay.T @ Ay)
            Lx_factor = splu(csc_matrix(LHSx))
            Ly_factor = splu(csc_matrix(LHSy))
            rho_cache = rho

        # --- RHS and solves in reduced variables xi ---
        RHSx = rho * (Ax.T @ (ZU[:, 0] - bx)) - fx
        RHSy = rho * (Ay.T @ (ZU[:, 1] - by)) - fy

        xi_x = Lx_factor.solve(RHSx)
        xi_y = Ly_factor.solve(RHSy)

        # --- Recover coefficients for this iterate ---
        a_x_stacked = coeffs_from_xi_x(xi_x)  # (6S,)
        a_y_stacked = coeffs_from_xi_y(xi_y)

        coeffs_x = [Const(a_x_stacked[6*i:6*(i+1)]) for i in range(S)]
        coeffs_y = [Const(a_y_stacked[6*i:6*(i+1)]) for i in range(S)]

        # --- Build x_traj at the ADMM sample points (m_per_seg per segment) ---
        x_traj = []
        for i in range(S):
            dt = segment_times[i+1] - segment_times[i]
            t_vals = np.linspace(0, dt, m_per_seg)
            ax_i = coeffs_x[i].value
            ay_i = coeffs_y[i].value
            xs = [evaluate_polynomial(ax_i, t) for t in t_vals]
            ys = [evaluate_polynomial(ay_i, t) for t in t_vals]
            x_traj.append(np.column_stack((xs, ys)))

        # --- Projection to convex regions at the same sampling density ---
        start_proj = time.perf_counter()
        projected_x, projected_y, reassigned = project_segments_to_convex_regions(
            coeffs_x, coeffs_y, segment_times, A_list, b_list, n_samples=m_per_seg
        )
        end_proj = time.perf_counter()
        z_traj = [np.column_stack((x, y)) for x, y in zip(projected_x, projected_y)]


        end_iter = time.perf_counter()



        x_traj = []
        for i in range(len(segment_times) - 1):
            t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], m_per_seg)
            a_x_seg = coeffs_x[i].value   # avoid shadowing stacked vector
            a_y_seg = coeffs_y[i].value
            x_vals = [evaluate_polynomial(a_x_seg, t) for t in t_vals]
            y_vals = [evaluate_polynomial(a_y_seg, t) for t in t_vals]
            x_traj.append(np.column_stack((x_vals, y_vals)))

        start_proj = time.perf_counter()
        projected_x, projected_y, reassigned = project_segments_to_convex_regions(
            coeffs_x, coeffs_y, segment_times, A_list, b_list, n_samples=m_per_seg
        )
        end_proj = time.perf_counter()


        z_traj = [np.column_stack((x, y)) for x, y in zip(projected_x, projected_y)]

        r_inf, s_inf = admm_residuals_stacked(
            Phi, a_x_stacked, a_y_stacked,
            z_traj, z_traj_prev, rho
        )


        if k % 1 == 0:
            print(f"resids: r_inf={r_inf:.3e}, s_inf={s_inf:.3e}, rho={rho:.3e}")

        # Dual update (scaled)
        u_traj = [u + (x - z) for u, x, z in zip(u_traj, x_traj, z_traj)]

        if k >= 3 and (k % 5 == 0):  # gate updates
            Z = np.vstack(z_traj)    # ((S*m) x 2)
            U = np.vstack(u_traj)    # ((S*m) x 2)

            rho_new = update_rho_osqp_with_s(
                rho,
                a_x_stacked, a_y_stacked,
                Phi, Z, U, Q_blk,
                s_inf,                        # from your admm_residuals_stacked
                rho_min=1e-6, rho_max=1e6,
                step_limit=5.0
            )
            if rho_new != rho:
                scale = rho / rho_new
                u_traj = [scale * u for u in u_traj]
                rho = rho_new
                rho_cache = None   # force refactorization with new rho

                


        rho_list.append(rho)

        # KONVERGENZTEST
        max_diff = max(np.linalg.norm(x - z, ord=np.inf) for x, z in zip(x_traj, z_traj))
        print(f"Max segment difference: {max_diff:.5f}")

        if max_diff < eps:
            print("Konvergenz erreicht.")
            ax = pdc.visualize_environment(Al=A_list, bl=b_list, p=path_real, planar=True)

            ax.plot(start_xy[0], start_xy[1], 'go', label='Start')
            ax.plot(goal_xy[0], goal_xy[1], 'bo', label='Goal')
            ax.plot(obstacles[:, 0], obstacles[:, 1], "o", color="green", label="Bäume")

            # Projizierte Segmente (rot)
            already_labeled = set()
            for x_vals, y_vals in zip(projected_x, projected_y):
                label = "projiziert" if "projiziert" not in already_labeled else ""
                ax.plot(x_vals, y_vals, 'r-', linewidth=2, label=label)
                already_labeled.add("projiziert")


            # Aktuelle Trajektorie (bunt)
            for segment, color in zip(x_traj, colors):
                ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=1.5)

            ax.set_title(f"ADMM Iteration {k+1}")
            ax.set_aspect('equal')
            ax.grid(True)
            ax.legend()
            plt.show()
            break

        ax = pdc.visualize_environment(Al=A_list, bl=b_list, p=path_real, planar=True)


        ax.plot(start_xy[0], start_xy[1], 'go', label='Start')
        ax.plot(goal_xy[0], goal_xy[1], 'bo', label='Goal')
        ax.plot(obstacles[:, 0], obstacles[:, 1], "o", color="green", label="Bäume")

        # Projizierte Segmente (rot)
        already_labeled = set()
        for x_vals, y_vals in zip(projected_x, projected_y):
            label = "projiziert" if "projiziert" not in already_labeled else ""
            ax.plot(x_vals, y_vals, 'r-', linewidth=2, label=label)
            already_labeled.add("projiziert")

        for idx, (x_vals, y_vals) in enumerate(zip(projected_x, projected_y)):
                if any(r[0] == idx for r in reassigned):
                    ax.plot(
                        x_vals, y_vals,
                        color="magenta", linewidth=4, linestyle="--", marker="o", markersize=8,
                        label="reassigned" if "reassigned" not in ax.get_legend_handles_labels()[1] else ""
                    )
                    print("should plot")

        # Aktuelle Trajektorie (bunt)
        for segment, color in zip(x_traj, colors):
            ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=1.5)

        ax.set_title(f"ADMM Iteration {k+1}")
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()

        iters = np.arange(len(rho_list))
        ax_rho.plot(iters, rho_list, 'k-', label="ρ")
        ax_rho.set_xlabel("Iteration")
        ax_rho.set_ylabel("ρ")
        ax_rho.grid(True)
        print(f"Primal {k+1} Dauer: {end_iter - start_iter:.7f} Sekunden")
        print(f"Projection {k+1} Dauer: {end_proj - start_proj:.7f} Sekunden")
        plt.pause(0.3)
        plt.clf()

    z_traj_prev = [z.copy() for z in z_traj]


