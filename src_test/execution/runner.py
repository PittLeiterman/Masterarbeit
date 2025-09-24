def run_admm_trajectory_optimization(config, DEBUG=False):    
    from input.make3DObstacles import load_voxel_grid   # or: from voxel_grid import load_voxel_grid
    from pathfinder.AStar3D import astar_3d
    from utils.path_manipulation import simplify_path_3d, keep_turns_np

    from optimization.primal_step import evaluate_polynomial

    from optimization.minco import precompute_mapping
    from utils.cvx_compat import Const
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    from utils.bernstein import build_T_block
    from optimization.projection_utils import project_segments_with_coverage
    

    import pydecomp as pdc
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import pyvista as pv

    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon as MplPolygon
    from math import comb

    import time

    def pv_surface_from_grid(mask3d: np.ndarray, origin=(0,0,0)):
        """
        mask3d: bool array (nx, ny, nz), True = occupied.
        Returns a PyVista surface extracted from the occupied volume.
        """
        nx, ny, nz = mask3d.shape
        ox, oy, oz = origin
        img = pv.ImageData(dimensions=(nx+1, ny+1, nz+1),
                        spacing=(1,1,1),
                        origin=(ox, oy, oz))
        # VTK expects cell data in Fortran order
        img.cell_data["occ"] = mask3d.astype(np.uint8).ravel(order="F")
        vol = img.threshold(0.5, scalars="occ")     # keep occupied cells
        return vol.extract_surface()                # outer surface only
    

    def visualize_voxelgrid_with_path(
        vg,
        path_idx=None,
        tube_radius=0.2,
        grid_opacity=0.25,
        grid_color="blue",
        turns_idx=None,              # list/array of (i,j,k) turn points
        mark_start_end=True,
        mark_turns=True,
        turn_color="orange",
        turn_scale=1.6,
        # --- NEW: convex decomposition visualization ---
        A_list=None,                 # list of (m_i, 3) arrays
        b_list=None,                 # list of (m_i, 1) or (m_i,) arrays
        show_decomposition=True,
        decomp_color="yellow",
        decomp_opacity=0.18,
        smooth_shading=True,
    ):
        """
        vg: VoxelGrid (with vg.grid bool and vg.info.origin, vg.info.to_coord(idx))
        path_idx: list/array of (i,j,k) indices (A* output) or None
        turns_idx: list/array of (i,j,k) turn points
        A_list, b_list: convex decomposition halfspaces per path segment
                        Each A is (m,3), b is (m,1) or (m,)
                        Inequality format returned by your binding: A x - b <= 0
        """
        surf = pv_surface_from_grid(vg.grid, origin=vg.info.origin)

        p = pv.Plotter()
        p.add_mesh(surf, color=grid_color, opacity=grid_opacity, show_edges=False)

        if path_idx is None:
            p.show_axes(); p.show(); return

        path_idx = np.asarray(path_idx)
        if path_idx.shape[0] >= 2:
            # Path as tube through voxel centers (world coords)
            centers = np.array([np.array(vg.info.to_coord(idx)) + 0.5 for idx in path_idx], float)

            line = pv.Spline(centers, n_points=len(centers))
            p.add_mesh(line.tube(radius=tube_radius, n_sides=16), color="red")

            if mark_start_end:
                p.add_mesh(pv.Sphere(radius=tube_radius*turn_scale, center=centers[0]),  color="green")
                p.add_mesh(pv.Sphere(radius=tube_radius*turn_scale, center=centers[-1]), color="orange")

            # Turn markers
            if mark_turns and turns_idx is not None:
                T = np.asarray(turns_idx)
                if T.ndim == 2 and T.shape[1] == 3 and T.shape[0] > 0:
                    T_centers = np.array([np.array(vg.info.to_coord(idx)) + 0.5 for idx in T], float)
                    pts = pv.PolyData(T_centers)
                    glyph_geom = pv.Sphere(radius=tube_radius*turn_scale)
                    glyphs = pts.glyph(scale=False, geom=glyph_geom)
                    p.add_mesh(glyphs, color=turn_color)

            print("[viz] surf points/cells:", getattr(surf, "n_points", None), getattr(surf, "n_cells", None))
            print("[viz] surf bounds:", getattr(surf, "bounds", None))
            print("[viz] centers min/max:", centers.min(axis=0), centers.max(axis=0))
            if A_list is not None:
                print("[viz] segments:", len(A_list))


            # ----------- NEW: draw convex decomposition polyhedra -----------
            if show_decomposition and A_list is not None and b_list is not None:
                try:
                    import cdd
                    from scipy.spatial import ConvexHull
                    have_cdd = True
                except Exception:
                    have_cdd = False

                # Fallback if cdd isn't available: keep your old HalfspaceIntersection route
                if not have_cdd:
                    try:
                        from scipy.spatial import HalfspaceIntersection, ConvexHull
                        have_hi = True
                    except Exception:
                        have_hi = False

                nseg = min(len(A_list), len(b_list), len(centers) - 1)
                for i in range(nseg):
                    A = np.asarray(A_list[i], dtype=float)
                    b = np.asarray(b_list[i], dtype=float).reshape(-1)

                    # --- 1) Skip ill-shaped entries
                    if A.ndim != 2 or A.shape[1] != 3 or b.ndim != 1 or b.size != A.shape[0]:
                        continue

                    # --- 2) Remove inactive/zero rows (critical!)
                    active = ~np.all(np.isclose(A, 0.0, atol=1e-12), axis=1)
                    A = A[active]
                    b = b[active]
                    if A.shape[0] < 4:
                        # Not enough planes to form a bounded 3D polyhedron
                        continue

                    if have_cdd:
                        # --- 3) Use cdd to convert H-rep (Ax - b <= 0) -> vertices
                        # cdd uses:  b - A x >= 0  ==> [b | -A] with INEQUALITY rep
                        try:
                            mat = cdd.Matrix(np.hstack([b[:, None], -A]), number_type='float')
                            mat.rep_type = cdd.RepType.INEQUALITY
                            poly = cdd.Polyhedron(mat)
                            gens = poly.get_generators()

                            # cdd returns points/rays; keep only points (first col == 1)
                            # and drop the leading "type" column to get xyz
                            G = np.array(gens, dtype=float)
                            if G.ndim != 2 or G.shape[1] < 4:
                                continue
                            # First column: 1 for point, 0 for ray (in standard cdd format)
                            is_point = np.isclose(G[:, 0], 1.0)
                            verts = G[is_point, 1:4]
                            if verts.shape[0] < 4:
                                continue

                            # --- 4) Triangulate the convex poly via ConvexHull
                            hull = ConvexHull(verts)
                            faces = []
                            for tri in hull.simplices:
                                faces.extend([3, int(tri[0]), int(tri[1]), int(tri[2])])  # pyvista face format

                            mesh = pv.PolyData(verts, faces)
                            p.add_mesh(
                                mesh,
                                color=decomp_color,
                                opacity=decomp_opacity,
                                smooth_shading=smooth_shading,
                            )

                        except Exception as e:
                            # Optional: print(e)
                            continue

                    elif have_hi:
                        # --- Fallback: HalfspaceIntersection (needs strictly interior point)
                        # Convert A x - b <= 0  ->  A x + c <= 0 with c = -b
                        hs = np.hstack([A, (-b)[:, None]])
                        # Use segment midpoint; may fail if not strictly interior
                        pt_inside = 0.5 * (centers[i] + centers[i + 1])
                        try:
                            hs_int = HalfspaceIntersection(hs, interior_point=pt_inside)
                            verts = np.asarray(hs_int.intersections)
                            if verts.shape[0] >= 4:
                                hull = ConvexHull(verts)
                                faces = []
                                for tri in hull.simplices:
                                    faces.extend([3, *tri.tolist()])
                                mesh = pv.PolyData(verts, faces)
                                p.add_mesh(mesh, color=decomp_color, opacity=decomp_opacity,
                                        smooth_shading=smooth_shading)
                        except Exception:
                            # silently skip on failure
                            continue
                    else:
                        # No cdd / no SciPy halfspaces available: nothing to draw
                        pass


        p.show_axes()
        p.show()
    


    
    def straight_line_path_3d(start_xyz, goal_xyz, num_nodes, center_offsets=False):
        """
        Return an (num_nodes, 3) straight line in 3D from start_xyz to goal_xyz.
        If center_offsets=True, adds +0.5 to each coordinate to hit voxel centers.
        """
        s = np.asarray(start_xyz, dtype=float)
        g = np.asarray(goal_xyz, dtype=float)
        if center_offsets:
            s = s + 0.5
            g = g + 0.5
        xs = np.linspace(s[0], g[0], num_nodes)
        ys = np.linspace(s[1], g[1], num_nodes)
        zs = np.linspace(s[2], g[2], num_nodes)
        return np.column_stack([xs, ys, zs])  # (num_nodes, 3)


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
    vg = load_voxel_grid(f"input/data/{shape}.txt", padding=0)
    grid3d = vg.grid  # bool (nx, ny, nz), True = occupied

    # start/goal are already 3D in your config: (x,y,z)
    start_idx = vg.info.to_index(start)
    goal_idx  = vg.info.to_index(goal)

    # quick sanity
    nx, ny, nz = vg.info.shape
    def inb(i,j,k): return (0<=i<nx and 0<=j<ny and 0<=k<nz)
    if not inb(*start_idx):
        raise ValueError(f"Start out of bounds: world={start}, idx={start_idx}, shape={vg.info.shape}")
    if not inb(*goal_idx):
        raise ValueError(f"Goal out of bounds: world={goal}, idx={goal_idx}, shape={vg.info.shape}")
    if grid3d[start_idx]:
        raise ValueError(f"Start is inside an obstacle: {start} (idx {start_idx})")
    if grid3d[goal_idx]:
        raise ValueError(f"Goal is inside an obstacle: {goal} (idx {goal_idx})")


    CONNECTIVITY = 6   # or 18 / 26 if you want diagonals
    path_idx = astar_3d(grid3d, start_idx, goal_idx, connectivity=CONNECTIVITY)


    if not path_idx:
        print("Kein Pfad gefunden!")
        visualize_voxelgrid_with_path(vg, path_idx=None, tube_radius=0.2)
        return  # or exit()


    print(f"Pfad gefunden")
    # Pfad Vereinfachung
    turns = np.asarray(keep_turns_np(path_idx))
    turns_idx = turns[keep]
    
    occ_idx   = np.argwhere(vg.grid == 1)            # (M,3)
    obs_np    = (occ_idx + 0.5).astype(np.float64)   # (M,3)

    # Path as voxel-center indices
    path_np   = (np.asarray(turns_idx, float) + 0.5)  # (K,3)

    # Local bbox in index units (size of a voxel == 1)
    box_np    = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)  # (1,3)


    # --- sanity (shapes/dtypes the C++ wants) ---
    assert obs_np.ndim == 2 and obs_np.shape[1] == 3, f"obs_np must be (N,3), got {obs_np.shape}"
    assert path_np.ndim == 2 and path_np.shape[1] == 3, f"path_np must be (N,3), got {path_np.shape}"
    assert box_np.shape == (1, 3), f"box_np must be (1,3), got {box_np.shape}"

    # --- call ---
    A_list, b_list = pdc.convex_decomposition_3D(obs_np, path_np, box_np)


    print("Konvexe Zerlegung abgeschlossen")


    # -------- (1) Knot positions: straight line from start -> goal --------
    S = int(num_segments)
    init_path = straight_line_path_3d(start_idx, goal_idx, S + 1, center_offsets=True)  # (S+1, 3)

    # Split per-axis if you still need individual arrays downstream
    p_all_x = init_path[:, 0]
    p_all_y = init_path[:, 1]
    p_all_z = init_path[:, 2]

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

        z_traj_prev = z_traj.copy()   
        end_iter = time.perf_counter()

        # Projektion pro Segment auf EIN Set (Kontrollpunkte)
        start_proj = time.perf_counter()
        C_segments = [X[ctrl_per_seg*i : ctrl_per_seg*(i+1), :] for i in range(S)]
        z_traj, assign, _ = project_segments_with_coverage(C_segments, A_list, b_list)

        end_proj = time.perf_counter()
    

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


