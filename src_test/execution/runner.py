def run_admm_trajectory_optimization(config, DEBUG=False):    
    from input.trees2D import create_occupancy_grid, load_forest_from_file
    from pathfinder.AStar import astar
    from utils.path_manipulation import simplify_path, upsample_path

    from optimization.primal_step import minimum_snap_trajectory, evaluate_polynomial, solve_primal_step
    from optimization.projection_utils import project_segments_to_convex_regions

    import pydecomp as pdc
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

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

    # Umgedreht (x, y) → für pydecomp & plotting
    start_xy = (start[1], start[0])
    goal_xy = (goal[1], goal[0])


    # A* Pfadsuche
    path = astar(grid, start, goal)

    # Pfad prüfen
    if path is None or len(path) == 0:
        print("Kein Pfad gefunden!")
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

        # Hindernisse (Wald)
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

    projected_x, projected_y = project_segments_to_convex_regions(
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

    for x_vals, y_vals in zip(projected_x, projected_y):
        ax.plot(x_vals, y_vals, 'r-', linewidth=2, label="projiziert" if 'projiziert' not in ax.get_legend_handles_labels()[1] else "")

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

    for k in range(max_iters):
        print(f"--- Iteration {k+1} ---")
        if k == 0:
            z_traj_prev = z_traj.copy()
        if k > 0:
            z_traj_extrapolated = [
                z_curr + beta * (z_curr - z_prev)
                for z_curr, z_prev in zip(z_traj, z_traj_prev)
            ]
        else:
            z_traj_extrapolated = z_traj

        psi = psi_iterating # Update psi for next iteration


        coeffs_x, coeffs_y = solve_primal_step(
            z_traj_extrapolated, u_traj, segment_times, start_xy, goal_xy, v_start, v_end, rho, psi, upsampled_path
        )

        # Neue x-Trajektorie berechnen (aus Polynomkoeffizienten)
        x_traj = []
        for i in range(len(segment_times) - 1):
            t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], num_segments)
            a_x = coeffs_x[i].value
            a_y = coeffs_y[i].value
            x_vals = [evaluate_polynomial(a_x, t) for t in t_vals]
            y_vals = [evaluate_polynomial(a_y, t) for t in t_vals]
            x_traj.append(np.column_stack((x_vals, y_vals)))

        projected_x, projected_y = project_segments_to_convex_regions(
            coeffs_x, coeffs_y, segment_times, A_list, b_list,
            path_reference=path_real,
            use_path_guidance=use_path_guidance,
            lambda_path=lambda_path,
            n_samples=num_segments
        )
        z_traj = [np.column_stack((x, y)) for x, y in zip(projected_x, projected_y)]

        # DUAL UPDATE
        u_traj = [u + (x - z) for u, x, z in zip(u_traj, x_traj, z_traj)]

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

        # OPTIONAL: Zwischenstand visualisieren
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
        plt.pause(0.1)
        plt.clf()

        

        z_traj_prev = z_traj.copy()


