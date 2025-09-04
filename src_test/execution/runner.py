def run_admm_trajectory_optimization(config, DEBUG=False):    
    from input.trees2D import create_occupancy_grid, load_forest_from_file
    from pathfinder.AStar import astar
    from utils.path_manipulation import simplify_path, upsample_path

    from optimization.primal_step import minimum_snap_trajectory, evaluate_polynomial, get_snap_cost_matrix
    from optimization.projection_utils import project_segments_to_convex_regions

    import pydecomp as pdc
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    import numpy as np
    import cvxpy as cp

    import time

    def precompute_bases(segment_times, n_samples_per_seg):
        """Precompute all time-basis rows, the design matrices Phi, mid rows, and snap Qs."""
        num_segments = len(segment_times) - 1
        Phi_list, Tmid_list, bounds, Q_list = [], [], [], []

        for i in range(num_segments):
            t0, t1 = segment_times[i], segment_times[i+1]
            dt = t1 - t0

            # Snap matrix (yours)
            Q_list.append(get_snap_cost_matrix(dt))

            # Uniform sample times inside [0, dt] with the same count you already use
            t_vals = np.linspace(0, dt, n_samples_per_seg)
            Phi = np.vstack([t_vals**k for k in range(6)]).T  # (m, 6)
            Phi_list.append(Phi)

            # Midpoint basis row
            tm = 0.5 * dt
            Tmid_list.append(np.array([1, tm, tm**2, tm**3, tm**4, tm**5]))

            # Boundary rows
            T_end     = np.array([1, dt, dt**2, dt**3, dt**4, dt**5])
            T_dot_end = np.array([0, 1, 2*dt, 3*dt**2, 4*dt**3, 5*dt**4])
            T_dd_end  = np.array([0, 0, 2, 6*dt, 12*dt**2, 20*dt**3])
            T_ddd_end = np.array([0, 0, 0, 6, 24*dt, 60*dt**2])

            T_start   = np.array([1, 0, 0, 0, 0, 0])
            T_dot0    = np.array([0, 1, 0, 0, 0, 0])
            T_ddot0   = np.array([0, 0, 2, 0, 0, 0])
            T_dddot0  = np.array([0, 0, 0, 6, 0, 0])

            bounds.append(dict(
                T_end=T_end, T_dot_end=T_dot_end, T_dd_end=T_dd_end, T_ddd_end=T_ddd_end,
                T_start=T_start, T_dot0=T_dot0, T_ddot0=T_ddot0, T_dddot0=T_dddot0
            ))

        return Phi_list, Tmid_list, bounds, Q_list


    def build_problem(segment_times, start_xy, goal_xy, v_start, v_end,
                    path_mid_targets,  # shape (num_segments, 2) midpoints of reference path
                    Phi_list, Tmid_list, bounds, Q_list):
        """Build the CVXPY problem once. Expose Parameters for (z - u), rho, psi."""
        num_segments = len(segment_times) - 1
        m = Phi_list[0].shape[0]

        # Variables per segment (6th-order poly in x and y)
        ax = [cp.Variable(6) for _ in range(num_segments)]
        ay = [cp.Variable(6) for _ in range(num_segments)]

        # Parameters that change each ADMM iteration
        rho = cp.Parameter(nonneg=True, value=1.0)
        psi = cp.Parameter(nonneg=True, value=1.0)
        Zu_x = [cp.Parameter(m, value=np.zeros(m)) for _ in range(num_segments)]
        Zu_y = [cp.Parameter(m, value=np.zeros(m)) for _ in range(num_segments)]

        # Cost
        cost_terms = []
        for i in range(num_segments):
            Q = Q_list[i]
            Phi = Phi_list[i]
            Tmid = Tmid_list[i]
            tx_i, ty_i = path_mid_targets[i]

            # Snap
            cost_terms += [cp.quad_form(ax[i], Q) + cp.quad_form(ay[i], Q)]
            # ADMM tracking of z - u
            cost_terms += [(rho/2) * cp.sum_squares(Phi @ ax[i] - Zu_x[i])]
            cost_terms += [(rho/2) * cp.sum_squares(Phi @ ay[i] - Zu_y[i])]
            # Path guidance at midpoint
            cost_terms += [psi * cp.sum_squares(Tmid @ ax[i] - tx_i)]
            cost_terms += [psi * cp.sum_squares(Tmid @ ay[i] - ty_i)]

        constraints = []
        # Start boundary
        T0    = np.array([1, 0, 0, 0, 0, 0])
        T0dot = np.array([0, 1, 0, 0, 0, 0])
        constraints += [
            ax[0] @ T0 == start_xy[0],
            ay[0] @ T0 == start_xy[1],
            ax[0] @ T0dot == v_start[0],
            ay[0] @ T0dot == v_start[1],
        ]

        # End boundary
        dt_end = segment_times[-1] - segment_times[-2]
        T1    = np.array([1, dt_end, dt_end**2, dt_end**3, dt_end**4, dt_end**5])
        T1dot = np.array([0, 1, 2*dt_end, 3*dt_end**2, 4*dt_end**3, 5*dt_end**4])
        constraints += [
            ax[-1] @ T1 == goal_xy[0],
            ay[-1] @ T1 == goal_xy[1],
            ax[-1] @ T1dot == v_end[0],
            ay[-1] @ T1dot == v_end[1],
        ]

        # C^3 continuity
        for i in range(num_segments - 1):
            b = bounds[i]
            constraints += [
                ax[i] @ b["T_end"]     == ax[i+1] @ b["T_start"],
                ay[i] @ b["T_end"]     == ay[i+1] @ b["T_start"],
                ax[i] @ b["T_dot_end"] == ax[i+1] @ b["T_dot0"],
                ay[i] @ b["T_dot_end"] == ay[i+1] @ b["T_dot0"],
                ax[i] @ b["T_dd_end"]  == ax[i+1] @ b["T_ddot0"],
                ay[i] @ b["T_dd_end"]  == ay[i+1] @ b["T_ddot0"],
                ax[i] @ b["T_ddd_end"] == ax[i+1] @ b["T_dddot0"],
                ay[i] @ b["T_ddd_end"] == ay[i+1] @ b["T_dddot0"],
            ]

        prob = cp.Problem(cp.Minimize(cp.sum(cost_terms)), constraints)

        # Expose handles you’ll update per iteration
        return dict(prob=prob, rho=rho, psi=psi, Zu_x=Zu_x, Zu_y=Zu_y, ax=ax, ay=ay, Phi_list=Phi_list)


    def admm_residuals(Phi_list, coeffs_x, coeffs_y, z_traj, z_traj_prev, rho):
        """Primal r = Phi a - z (in sample space); Dual s = rho * Phi^T (z - z_prev) (in coeff space)."""
        r_inf = 0.0
        s_inf = 0.0
        for i, Phi in enumerate(Phi_list):
            ax_i = coeffs_x[i].value
            ay_i = coeffs_y[i].value
            # primal residuals at samples
            rx = Phi @ ax_i - z_traj[i][:, 0]
            ry = Phi @ ay_i - z_traj[i][:, 1]
            r_inf = max(r_inf, np.linalg.norm(rx, np.inf), np.linalg.norm(ry, np.inf))
            # dual residuals mapped back to coeff space
            dz = z_traj[i] - z_traj_prev[i]
            sx = rho * (Phi.T @ dz[:, 0])
            sy = rho * (Phi.T @ dz[:, 1])
            s_inf = max(s_inf, np.linalg.norm(sx, np.inf), np.linalg.norm(sy, np.inf))
        return r_inf, s_inf

    def update_rho(rho, r_inf, s_inf,
                tau_inc=2.0, kappa_inc=1.0, factor_inc=10.0,
                tau_dec=50.0, kappa_dec=0.2, factor_dec=10.0,
                rho_min=1e-6, rho_max=1e6):
        """
        Asymmetrisches rho-Update
        """

        if s_inf < 1e-16 and r_inf < 1e-16:
            return rho

        # Erhöhen (r >> s)
        if r_inf > tau_inc * s_inf and s_inf > 0:
            rho_new = rho * (r_inf / s_inf) ** kappa_inc * factor_inc

        # Reduzieren (s >> r)
        elif s_inf > tau_dec * r_inf and r_inf > 0:
            rho_new = rho * (r_inf / s_inf) ** kappa_dec * factor_dec

        else:
            rho_new = rho

        return float(np.clip(rho_new, rho_min, rho_max))



    # Parameter
    area_size = tuple(config["area_size"])
    shape = config["shape"]
    start = tuple(config["start"])
    goal = tuple(config["goal"])
    keep = config["keep"]
    rho = config["rho"]
    max_iters = config["max_iters"]
    eps = config["eps"]
    beta = config["beta"]
    v_start = tuple(config["v_start"])
    v_end = tuple(config["v_end"])
    use_path_guidance = config.get("use_path_guidance")
    lambda_path = config.get("lambda_path")
    num_segments = config.get("num_segments")
    psi = config.get("psi")
    psi_iterating = config.get("psi_iterating")


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

    upsampled_path = upsample_path(path_real, num_segments+1)

    # Generate trajectory coefficients
    coeffs_x, coeffs_y, segment_times = minimum_snap_trajectory(start_xy, goal_xy, v_start, v_end, upsampled_path, psi, num_segments=num_segments)

    Phi_list, Tmid_list, bounds, Q_list = precompute_bases(segment_times, n_samples_per_seg=num_segments)

    # Constant midpoint targets from the *upsampled* path (matches num_segments)
    mid_targets = []
    for i in range(len(segment_times) - 1):
        tx = 0.5 * (upsampled_path[i, 0] + upsampled_path[i+1, 0])
        ty = 0.5 * (upsampled_path[i, 1] + upsampled_path[i+1, 1])
        mid_targets.append((tx, ty))
    mid_targets = np.array(mid_targets)


    QP = build_problem(
        segment_times=segment_times,
        start_xy=start_xy, goal_xy=goal_xy,
        v_start=v_start, v_end=v_end,
        path_mid_targets=mid_targets,
        Phi_list=Phi_list, Tmid_list=Tmid_list, bounds=bounds, Q_list=Q_list
    )

    QP["rho"].value = rho

    projected_x, projected_y, reassigned = project_segments_to_convex_regions(
            coeffs_x, coeffs_y, segment_times, A_list, b_list,
            path_reference=path_real,
            use_path_guidance=use_path_guidance,
            lambda_path=lambda_path,
            n_samples=num_segments
        )

    colors = cm.viridis(np.linspace(0, 1, len(segment_times) - 1))

    segment_lengths = []

    for i in range(len(segment_times) - 1):
        t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], num_segments)
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
        t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], num_segments)
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

        if k == 0:
            z_traj_prev = [z.copy() for z in z_traj]

        
        if k > 0:
            z_traj_extrapolated = [z + beta * (z - zp) for z, zp in zip(z_traj, z_traj_prev)]
        else:
            z_traj_extrapolated = z_traj

        psi = psi_iterating
        QP["rho"].value = rho
        QP["psi"].value = psi

        for i in range(len(segment_times) - 1):
            Zu = z_traj_extrapolated[i] - u_traj[i]      # shape (m,2)
            QP["Zu_x"][i].value = Zu[:, 0]
            QP["Zu_y"][i].value = Zu[:, 1]

        QP["prob"].solve(solver=cp.OSQP, warm_start=True, verbose=False)
        coeffs_x = QP["ax"]
        coeffs_y = QP["ay"]
        end_iter = time.perf_counter()

        x_traj = []
        for i in range(len(segment_times) - 1):
            t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], num_segments)
            a_x = coeffs_x[i].value
            a_y = coeffs_y[i].value
            x_vals = [evaluate_polynomial(a_x, t) for t in t_vals]
            y_vals = [evaluate_polynomial(a_y, t) for t in t_vals]
            x_traj.append(np.column_stack((x_vals, y_vals)))

        projected_x, projected_y, reassigned = project_segments_to_convex_regions(
            coeffs_x, coeffs_y, segment_times, A_list, b_list,
            path_reference=path_real, use_path_guidance=use_path_guidance,
            lambda_path=lambda_path, n_samples=num_segments
        )
        z_traj = [np.column_stack((x, y)) for x, y in zip(projected_x, projected_y)]

        r_inf, s_inf = admm_residuals(QP["Phi_list"], coeffs_x, coeffs_y, z_traj, z_traj_prev, rho)
        if k % 1 == 0:
            print(f"resids: r_inf={r_inf:.3e}, s_inf={s_inf:.3e}, rho={rho:.3e}")

        # Dual update (scaled)
        u_traj = [u + (x - z) for u, x, z in zip(u_traj, x_traj, z_traj)]

        if k >= 3 and (k % 5 == 0):        # gate updates
            rho_new = update_rho(rho, r_inf, s_inf)
            if rho_new != rho:
                scale = rho / rho_new      
                u_traj = [scale * u for u in u_traj]
                rho = rho_new
                QP["rho"].value = rho

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
        print(f"Iter {k+1} Dauer: {end_iter - start_iter:.3f} Sekunden")
        plt.pause(0.1)
        plt.clf()

    z_traj_prev = [z.copy() for z in z_traj]


