def run_admm_trajectory_optimization(config, DEBUG=False):    
    from input.trees2D import create_occupancy_grid, load_forest_from_file
    from pathfinder.AStar import astar
    from utils.path_manipulation import simplify_path

    from optimization.primal_step import evaluate_polynomial

    from optimization.minco import precompute_mapping
    from optimization.sampling import build_Phi
    from utils.cvx_compat import Const
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    from utils.bernstein import build_T_block
    from optimization.projection_utils import project_segments_with_coverage
    

    import pydecomp as pdc
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon as MplPolygon
    from math import comb

    import time

    
    def straight_line_path(start_xy, goal_xy, num_nodes):
        xs = np.linspace(start_xy[0], goal_xy[0], num_nodes)
        ys = np.linspace(start_xy[1], goal_xy[1], num_nodes)
        return np.column_stack([xs, ys])  # (num_nodes, 2)

    def bernstein_basis_row(n, u):
        um = 1.0 - u
        return np.array([comb(n, k) * (um**(n-k)) * (u**k) for k in range(n+1)], dtype=float)

    def bezier_curve_from_cpoints(C_seg, res=800):
        """
        Erzeugt eine Bézier-Kurve aus Kontrollpunkten.
        C_seg: (n+1, 2) Kontrollpunkte, z. B. (6,2) für Quintic
        res: Anzahl Samples entlang der Kurve
        """
        C = np.asarray(C_seg, float).reshape(-1, 2)
        n = C.shape[0] - 1
        u = np.linspace(0.0, 1.0, res)
        pts = np.empty((res, 2))
        for i, ui in enumerate(u):
            B = bernstein_basis_row(n, ui)
            pts[i] = B @ C
        return pts[:,0], pts[:,1]



    def admm_residuals_cp(Acx, Acy, xi_x, xi_y, Z, Z_prev, rho_val, bcx, bcy):
        """
        Residuen für Nebenbedingung:  (T_blk @ (M xi + c)) = Z
        d.h.  Acx xi + bcx = Z  und  Acy xi + bcy = Z
        """
        xi_x = np.asarray(xi_x).ravel()
        xi_y = np.asarray(xi_y).ravel()

        # Primal residual r = A xi + b - Z
        rx = Acx @ xi_x + bcx - Z[:, 0]
        ry = Acy @ xi_y + bcy - Z[:, 1]
        r_inf = max(np.linalg.norm(rx, np.inf), np.linalg.norm(ry, np.inf))

        # Dual residual s = rho * A^T (Z - Z_prev)
        dZ = Z - Z_prev
        sx = rho_val * (Acx.T @ dZ[:, 0])
        sy = rho_val * (Acy.T @ dZ[:, 1])
        s_inf = max(np.linalg.norm(sx, np.inf), np.linalg.norm(sy, np.inf))
        return r_inf, s_inf







    def update_rho_osqp_with_s_cp(rho, r_inf, s_inf,
                              rho_min=1e-6, rho_max=1e6,
                              step_limit=5.0, eps=1e-12):
        """
        Skaliert rho nach der OSQP-Heuristik:
            rho <- rho * sqrt(||r|| / ||s||)
        Hier arbeiten wir direkt mit den Residuen, ohne Matrixprodukte.
        """
        if r_inf < eps and s_inf < eps:
            return rho
        scale = np.sqrt(r_inf / max(s_inf, eps))
        scale = float(np.clip(scale, 1.0/step_limit, step_limit))
        return float(np.clip(rho * scale, rho_min, rho_max))







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
    T_blk = build_T_block(segment_times, degree=5)

    # Reduced snap terms: H = M^T (2Q) M,  f = M^T (2Q) c
    Hx = (Mx.T @ (Q_blk @ Mx))
    fx = (Mx.T @ (Q_blk @ cx))
    Hy = (My.T @ (Q_blk @ My))
    fy = (My.T @ (Q_blk @ cy))

    Acx = T_blk @ Mx;  bcx = T_blk @ cx
    Acy = T_blk @ My;  bcy = T_blk @ cy

    ctrl_per_seg   = T_blk.shape[0] // S
    coeffs_per_seg = Mx.shape[0] // S 


    # Build initial coefficients (instead of CVXPY/solve_axis)
    # Startkoeffizienten aus xi
    a_x_stacked = coeffs_from_xi_x(xi_x)  # (6S,)
    a_y_stacked = coeffs_from_xi_y(xi_y)

    # Kontrollpunkte aus a (Power->Bernstein via T_blk)
    Cx0 = T_blk @ a_x_stacked
    Cy0 = T_blk @ a_y_stacked
    X = np.column_stack([Cx0, Cy0])          # ((6S) x 2)

    # Segmentweise Projektion der Kontrollpunkte in den "besten" Set
    C_segments = [X[ctrl_per_seg*i : ctrl_per_seg*(i+1), :] for i in range(S)]
    z_traj, assign, costs_mat = project_segments_with_coverage(C_segments, A_list, b_list)
    u_traj = np.zeros_like(z_traj)

    # (optional) Debug: Coverage prüfen
    counts = np.bincount(assign, minlength=len(A_list))
    print("Region-Segment-Zuordnung:", assign)
    print("Segmente je Region:", counts)


    z_traj_prev = z_traj.copy()

    # Farben fürs Plotten belassen
    colors = cm.viridis(np.linspace(0, 1, S))


    rho_list = []

        # ---- Einmalige Visualisierung vor ADMM ----
    ax0 = pdc.visualize_environment(Al=A_list, bl=b_list, p=path_real, planar=True)
    ax0.plot(start_xy[0], start_xy[1], 'go', label='Start')
    ax0.plot(goal_xy[0], goal_xy[1], 'bo', label='Goal')
    ax0.plot(obstacles[:, 0], obstacles[:, 1], "o", color="green", label="Bäume")

    face_alpha   = 0.12
    edge_lw      = 1.2
    curve_lw     = 2.0
    res_curve    = 900

    for i in range(S):
        C = z_traj[ctrl_per_seg*i : ctrl_per_seg*(i+1), :]

        # Konvexe Hülle
        try:
            hull = ConvexHull(C)
            H = C[hull.vertices]
        except Exception:
            H = C
        ax0.add_patch(MplPolygon(
            H, closed=True, facecolor='red', edgecolor='red',
            linewidth=edge_lw, alpha=face_alpha, zorder=1
        ))

        # Bézier-Kurve aus projizierten CPs
        bx, by = bezier_curve_from_cpoints(C, res=res_curve)
        ax0.plot(bx, by, 'r-', linewidth=curve_lw, alpha=0.95,
                label='Bézier (projiziert)' if i==0 else "", zorder=2)


    # Trajektorie zum Anschauen sampeln (nur Plot)
    coeffs_x = [Const(a_x_stacked[coeffs_per_seg*i : coeffs_per_seg*(i+1)]) for i in range(S)]
    coeffs_y = [Const(a_y_stacked[coeffs_per_seg*i : coeffs_per_seg*(i+1)]) for i in range(S)]
    for i in range(S):
        dt = segment_times[i+1] - segment_times[i]
        t_vals = np.linspace(0, dt, 100)
        ax_i = coeffs_x[i].value
        ay_i = coeffs_y[i].value
        xs = [evaluate_polynomial(ax_i, t) for t in t_vals]
        ys = [evaluate_polynomial(ay_i, t) for t in t_vals]
        ax0.plot(xs, ys, color='k', alpha=0.6, linewidth=1.5, label='Init curve' if i==0 else "")

    ax0.set_aspect('equal'); ax0.grid(True); ax0.legend()
    plt.show()

    # --- Rho-Plot vorbereiten ---
    fig_rho, ax_rho = plt.subplots(figsize=(6, 3))
    ax_rho.set_xlabel("Iteration")
    ax_rho.set_ylabel("ρ")
    ax_rho.grid(True)

 
    for k in range(max_iters):
        print(f"--- Iteration {k+1} ---")
        start_iter = time.perf_counter()

        # --- Build ZU (same as before) ---
        if k == 0:
            z_traj_prev = z_traj.copy()

        ZU = z_traj - u_traj

        # --- Cache factorization when rho is unchanged ---
        if k == 0:
            rho_cache = None
        if (k == 0) or (rho_cache is None) or (abs(rho_cache - rho) > 0):
            LHSx = Hx + rho * (Acx.T @ Acx)
            LHSy = Hy + rho * (Acy.T @ Acy)
            Lx_factor = splu(csc_matrix(LHSx))
            Ly_factor = splu(csc_matrix(LHSy))
            rho_cache = rho

        # --- RHS and solves in reduced variables xi ---
        RHSx = rho * (Acx.T @ (ZU[:, 0] - bcx)) - fx
        RHSy = rho * (Acy.T @ (ZU[:, 1] - bcy)) - fy

        xi_x = Lx_factor.solve(RHSx)
        xi_y = Ly_factor.solve(RHSy)

        # --- Recover coefficients for this iterate ---
        a_x_stacked = coeffs_from_xi_x(xi_x)  # (6S,)
        a_y_stacked = coeffs_from_xi_y(xi_y)

        coeffs_x = [Const(a_x_stacked[coeffs_per_seg*i : coeffs_per_seg*(i+1)]) for i in range(S)]
        coeffs_y = [Const(a_y_stacked[coeffs_per_seg*i : coeffs_per_seg*(i+1)]) for i in range(S)]



        # Aktuelle Kontrollpunkte X (aus a)
        Cx = T_blk @ a_x_stacked
        Cy = T_blk @ a_y_stacked
        X  = np.column_stack([Cx, Cy])          # ((6S) x 2)

        # Projektion pro Segment auf EIN Set (Kontrollpunkte)
        start_proj = time.perf_counter()
        C_segments = [X[ctrl_per_seg*i : ctrl_per_seg*(i+1), :] for i in range(S)]
        z_traj, assign, _ = project_segments_with_coverage(C_segments, A_list, b_list)

        end_proj = time.perf_counter()

        z_traj_prev = z_traj.copy()          # ((6S) x 2)



        end_iter = time.perf_counter()



        x_traj = []
        for i in range(len(segment_times) - 1):
            t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], m_per_seg)
            a_x_seg = coeffs_x[i].value   # avoid shadowing stacked vector
            a_y_seg = coeffs_y[i].value
            x_vals = [evaluate_polynomial(a_x_seg, t) for t in t_vals]
            y_vals = [evaluate_polynomial(a_y_seg, t) for t in t_vals]
            x_traj.append(np.column_stack((x_vals, y_vals)))


        r_inf, s_inf = admm_residuals_cp(Acx, Acy, xi_x, xi_y, z_traj, z_traj_prev, rho, bcx, bcy)


        if k % 1 == 0:
            print(f"resids: r_inf={r_inf:.3e}, s_inf={s_inf:.3e}, rho={rho:.3e}")

        # Dual update (scaled)
        u_traj = u_traj + (X - z_traj)

        if k >= 3 and (k % 5 == 0):  # gate updates
            rho_new = update_rho_osqp_with_s_cp(
                rho, r_inf, s_inf,
                rho_min=1e-6, rho_max=1e6, step_limit=5.0
            )
            if rho_new != rho:
                scale = rho / rho_new
                u_traj = scale * u_traj
                rho = rho_new
                rho_cache = None


        rho_list.append(rho)

        # KONVERGENZTEST
        max_diff = float(np.max(np.abs(X - z_traj)))
        print(f"Max segment difference: {max_diff:.5f}")

        if max_diff < eps:
            print("Konvergenz erreicht.")
            ax = pdc.visualize_environment(Al=A_list, bl=b_list, p=path_real, planar=True)

            ax.plot(start_xy[0], start_xy[1], 'go', label='Start')
            ax.plot(goal_xy[0], goal_xy[1], 'bo', label='Goal')
            ax.plot(obstacles[:, 0], obstacles[:, 1], "o", color="green", label="Bäume")

            
            # Visualisiere projizierte Kontrollpunkte (Convex-Hull sichtbar)
            for i in range(S):
                C = z_traj[ctrl_per_seg*i : ctrl_per_seg*(i+1), :]

                try:
                    hull = ConvexHull(C)
                    H = C[hull.vertices]
                except Exception:
                    H = C
                ax.add_patch(MplPolygon(
                    H, closed=True, facecolor='red', edgecolor='red',
                    linewidth=edge_lw, alpha=face_alpha, zorder=1
                ))

                bx, by = bezier_curve_from_cpoints(C, res=res_curve)
                ax.plot(bx, by, 'r-', linewidth=curve_lw, alpha=0.95,
                        label='Bézier (projiziert)' if i==0 else "", zorder=2)




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

        # Visualisiere projizierte Kontrollpunkte (Convex-Hull sichtbar)
        face_alpha   = 0.12
        edge_lw      = 1.2
        curve_lw     = 2.0
        res_curve    = 900

        for i in range(S):
            C = z_traj[ctrl_per_seg*i : ctrl_per_seg*(i+1), :]

            try:
                hull = ConvexHull(C)
                H = C[hull.vertices]
            except Exception:
                H = C
            ax.add_patch(MplPolygon(
                H, closed=True, facecolor='red', edgecolor='red',
                linewidth=edge_lw, alpha=face_alpha, zorder=1
            ))

            bx, by = bezier_curve_from_cpoints(C, res=res_curve)
            ax.plot(bx, by, 'r-', linewidth=curve_lw, alpha=0.95,
                    label='Bézier (projiziert)' if i==0 else "", zorder=2)


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

    z_traj_prev = z_traj.copy()


