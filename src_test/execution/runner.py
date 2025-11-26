def run_admm_trajectory_optimization(config, DEBUG=False, PROJECTIONS=False):    
    from input.make3DObstacles import load_voxel_grid
    from pathfinder.AStar3D import astar_3d
    from optimization.primal_step import evaluate_polynomial
    from optimization.minco import precompute_mapping
    from optimization.projection_utils import project_segments_with_coverage, prepare_halfspaces
    
    from utils.path_manipulation import keep_turns_np, reduce_turns_by_los, sample_every_k
    from utils.bernstein import build_T_block
    from utils.cvx_compat import Const

    import os
    import csv
    import pydecomp as pdc
    import numpy as np
    import pyvista as pv
    import time

    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    def pv_surface_from_grid(mask3d: np.ndarray, origin=(0,0,0)):
        nx, ny, nz = mask3d.shape
        ox, oy, oz = origin
        img = pv.ImageData(dimensions=(nx+1, ny+1, nz+1),
                        spacing=(1,1,1),
                        origin=(ox, oy, oz))
        
        img.cell_data["occ"] = mask3d.astype(np.uint8).ravel(order="F")
        vol = img.threshold(0.5, scalars="occ")
        return vol.extract_surface()
    

    def visualize_voxelgrid_with_path(
        vg,
        path_idx=None,
        path_xyz=None,
        plotPath=True,
        tube_radius=0.2,
        grid_opacity=0.20,
        grid_color="blue",
        turns_idx=None,
        mark_start_end=True,
        mark_turns=True,
        turn_color="orange",
        turn_scale=3,
        A_list=None,
        b_list=None,
        show_decomposition=True,
        decomp_color="yellow",
        decomp_opacity=0.18,
        smooth_shading=True,
        filename = "start",
    ):
        surf = pv_surface_from_grid(vg.grid, origin=vg.info.origin)

        p = pv.Plotter()
        # p.add_mesh(surf, color=grid_color, opacity=grid_opacity, show_edges=False)

        if path_idx is None and path_xyz is None:
            p.show_axes(); p.show(); return

        centers = None
        if path_xyz is not None:
            centers = np.asarray(path_xyz, float)
        elif path_idx is not None:
            path_idx = np.asarray(path_idx)
            if path_idx.shape[0] >= 2:
                centers = np.array([np.array(vg.info.to_coord(idx)) + 0.5
                                    for idx in path_idx], float)

        if centers is not None and centers.shape[0] >= 2:
            line = pv.Spline(centers, n_points=len(centers))
            if plotPath:
                p.add_mesh(line.tube(radius=tube_radius, n_sides=16), color="red")

            if mark_start_end:
                p.add_mesh(pv.Sphere(radius=tube_radius*turn_scale, center=centers[0]),  color="green")
                p.add_mesh(pv.Sphere(radius=tube_radius*turn_scale, center=centers[-1]), color="orange")
    

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


            if show_decomposition and A_list is not None and b_list is not None:
                try:
                    import cdd
                    from scipy.spatial import ConvexHull
                    have_cdd = True
                except Exception:
                    have_cdd = False

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

                    if A.ndim != 2 or A.shape[1] != 3 or b.ndim != 1 or b.size != A.shape[0]:
                        continue

                    active = ~np.all(np.isclose(A, 0.0, atol=1e-12), axis=1)
                    A = A[active]
                    b = b[active]
                    if A.shape[0] < 4:
                        continue

                    if have_cdd:
                        try:
                            mat = cdd.Matrix(np.hstack([b[:, None], -A]), number_type='float')
                            mat.rep_type = cdd.RepType.INEQUALITY
                            poly = cdd.Polyhedron(mat)
                            gens = poly.get_generators()
                            G = np.array(gens, dtype=float)
                            if G.ndim != 2 or G.shape[1] < 4:
                                continue
                            is_point = np.isclose(G[:, 0], 1.0)
                            verts = G[is_point, 1:4]
                            if verts.shape[0] < 4:
                                continue

                            hull = ConvexHull(verts)
                            faces = []
                            for tri in hull.simplices:
                                faces.extend([3, int(tri[0]), int(tri[1]), int(tri[2])])

                            mesh = pv.PolyData(verts, faces)
                            p.add_mesh(
                                mesh,
                                color=decomp_color,
                                opacity=decomp_opacity,
                                smooth_shading=smooth_shading,
                            )

                        except Exception as e:
                            print(e)
                            continue

                    elif have_hi:
                        hs = np.hstack([A, (-b)[:, None]])
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
                            continue
                    else:
                        pass

        p.open_movie("3Dplots/"+ filename +".mp4", framerate=30)
        p.show(auto_close=False)

        n_frames = 360
        step = 360.0 / n_frames

        for i in range(n_frames):
            p.camera.azimuth += step
            p.render()
            p.write_frame()

        p.close()
    

    
    def straight_line_path_3d(start_xyz, goal_xyz, num_nodes, center_offsets=False):
        s = np.asarray(start_xyz, dtype=float)
        g = np.asarray(goal_xyz, dtype=float)
        if center_offsets:
            s = s + 0.5
            g = g + 0.5
        xs = np.linspace(s[0], g[0], num_nodes)
        ys = np.linspace(s[1], g[1], num_nodes)
        zs = np.linspace(s[2], g[2], num_nodes)
        return np.column_stack([xs, ys, zs])

    def admm_residuals_cp_3d_from_C(Cx, Cy, Cz, Z, Z_prev, Acx, Acy, Acz, rho):
        # Primal
        rx = Cx - Z[:, 0]
        ry = Cy - Z[:, 1]
        rz = Cz - Z[:, 2]
        r_inf = max(np.linalg.norm(rx, np.inf), np.linalg.norm(ry, np.inf), np.linalg.norm(rz, np.inf))

        # Dual
        dZ = Z - Z_prev
        sx = rho * (Acx.T @ dZ[:, 0])
        sy = rho * (Acy.T @ dZ[:, 1])
        sz = rho * (Acz.T @ dZ[:, 2])
        s_inf = max(np.linalg.norm(sx, np.inf), np.linalg.norm(sy, np.inf), np.linalg.norm(sz, np.inf))

        # Skalen
        scale_pri  = max(np.linalg.norm(Cx, np.inf), np.linalg.norm(Cy, np.inf),
                        np.linalg.norm(Cz, np.inf), np.max(np.abs(Z)))
        scale_dual = s_inf

        return r_inf, s_inf, scale_pri, scale_dual




    def update_rho_osqp_with_s_cp(rho, r_inf, s_inf,
                              rho_min=1e-6, rho_max=1e6,
                              step_limit=5.0, eps=1e-12):
        if r_inf < eps and s_inf < eps:
            return rho
        scale = np.sqrt(r_inf / max(s_inf, eps))
        scale = float(np.clip(scale, 1.0/step_limit, step_limit))
        return float(np.clip(rho * scale, rho_min, rho_max))


    # Parameter
    shape = config["shape"]
    start = tuple(config["start"])
    goal = tuple(config["goal"])
    rho = config["rho"]
    max_iters = config["max_iters"]
    v_start = tuple(config["v_start"])
    v_end = tuple(config["v_end"])
    num_segments = config.get("num_segments")
    segment_ratio = config.get("segment_ratio")
    m_per_seg = int(config.get("m_per_seg", 10))
    runtime_summary_csv = config.get("runtime_summary_csv", "results/runtime_summary.csv")
    eps_abs_pri  = float(config.get("eps_abs_pri", 1e-4))
    eps_abs_dual = float(config.get("eps_abs_dual", 1e-4))
    eps_rel      = float(config.get("eps_rel",     1e-3))
    corners = int(config["corners"])

    os.makedirs(os.path.dirname(runtime_summary_csv), exist_ok=True)
    runtime_rows = []

    summary_meta = {
        "shape": shape,
        "start": start,
        "goal": goal,
        "num_segments_cfg": int(num_segments),
        "m_per_seg": int(m_per_seg),
        "rho_init": float(rho),
        "max_iters": int(max_iters),
    }

    vg = load_voxel_grid(f"input/data/{shape}.txt", padding=0)
    grid3d = vg.grid

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


    CONNECTIVITY = 18   #6 / 18 / 26
    path_idx = astar_3d(grid3d, start_idx, goal_idx, connectivity=CONNECTIVITY)


    if not path_idx:
        print("Kein Pfad gefunden!")
        return


    print(f"Pfad gefunden")
    if corners != 0:
        turns_idx = sample_every_k(path_idx, corners)
    else:
        turns = np.asarray(keep_turns_np(path_idx))
        turns_idx = reduce_turns_by_los(turns, grid3d, clearance=0)

    if DEBUG:
        visualize_voxelgrid_with_path(
            vg,
            path_idx=path_idx,
            plotPath=True,
            tube_radius=0.25,
            grid_opacity=0.25,
            grid_color="blue",
            turns_idx=turns_idx,
            mark_start_end=True,
            mark_turns=True,
            A_list=None,
            b_list=None,
            show_decomposition=False,
            decomp_color="yellow",
            decomp_opacity=0.18,
            smooth_shading=True,
            filename="debug_astar_path"
        )
        exit()

    origin = np.asarray(vg.info.origin, dtype=float)
    occ_idx   = np.argwhere(vg.grid == 1)
    obs_np    = (occ_idx + 0.5).astype(np.float64) + origin
    path_np   = (np.asarray(turns_idx, float) + 0.5) + origin 
    box_np    = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)


    #sanity
    assert obs_np.ndim == 2 and obs_np.shape[1] == 3, f"obs_np must be (N,3), got {obs_np.shape}"
    assert path_np.ndim == 2 and path_np.shape[1] == 3, f"path_np must be (N,3), got {path_np.shape}"
    assert box_np.shape == (1, 3), f"box_np must be (1,3), got {box_np.shape}"

    A_list, b_list = pdc.convex_decomposition_3D(obs_np, path_np, box_np)

    print("Konvexe Zerlegung abgeschlossen")

    num_polytopes = int(min(len(A_list), len(b_list)))

    S = max(num_segments, int(num_polytopes * segment_ratio))
    init_path = straight_line_path_3d(start_idx, goal_idx, S + 1, center_offsets=True) + origin  # (S+1, 3)


    summary_meta.update({
        "num_polytopes": num_polytopes,
        "ratio": segment_ratio,
    })

    p_all_x = init_path[:, 0]
    p_all_y = init_path[:, 1]
    p_all_z = init_path[:, 2]

    def allocate_times_from_chords(path_xy, v_des=1.0, t_min=0.05):
        p = np.asarray(path_xy, float)
        chords = np.linalg.norm(np.diff(p, axis=0), axis=1)
        v_des = max(float(v_des), 1e-6)
        T_i = np.maximum(chords / v_des, float(t_min))
        return np.concatenate(([0.0], np.cumsum(T_i)))

    segment_times = allocate_times_from_chords(
        init_path,
        v_des=float(config.get("v_des", 1.0)),
        t_min=float(config.get("t_min", 0.05))
    )
    S = len(segment_times) - 1

    coeffs_from_xi_x, Mx, cx, Q_blk = precompute_mapping(
        segment_times,
        p0=float(p_all_x[0]), pS=float(p_all_x[-1]),
        v_start=float(v_start[0]), v_end=float(v_end[0])
    )

    coeffs_from_xi_y, My, cy, _ = precompute_mapping(
        segment_times,
        p0=float(p_all_y[0]), pS=float(p_all_y[-1]),
        v_start=float(v_start[1]), v_end=float(v_end[1])
    )

    coeffs_from_xi_z, Mz, cz, _ = precompute_mapping(
        segment_times,
        p0=float(p_all_z[0]), pS=float(p_all_z[-1]),
        v_start=float(v_start[2]), v_end=float(v_end[2])
    )

    xi_x = p_all_x[1:-1].copy()
    xi_y = p_all_y[1:-1].copy()
    xi_z = p_all_z[1:-1].copy()

    T_blk = build_T_block(segment_times, degree=5)

    Hx = Mx.T @ (Q_blk @ Mx)
    fx = Mx.T @ (Q_blk @ cx)

    Hy = My.T @ (Q_blk @ My)
    fy = My.T @ (Q_blk @ cy)

    Hz = Mz.T @ (Q_blk @ Mz)
    fz = Mz.T @ (Q_blk @ cz)

    Acx = T_blk @ Mx;  bcx = T_blk @ cx
    Acy = T_blk @ My;  bcy = T_blk @ cy
    Acz = T_blk @ Mz;  bcz = T_blk @ cz

    ctrl_per_seg   = T_blk.shape[0] // S
    coeffs_per_seg = Mx.shape[0] // S

    #Build initial coefficients
    a_x_stacked = coeffs_from_xi_x(xi_x)
    a_y_stacked = coeffs_from_xi_y(xi_y)
    a_z_stacked = coeffs_from_xi_z(xi_z)

    Cx0 = T_blk @ a_x_stacked
    Cy0 = T_blk @ a_y_stacked
    Cz0 = T_blk @ a_z_stacked

    X = np.column_stack([Cx0, Cy0, Cz0])
    Ab = prepare_halfspaces(A_list, b_list)
    C_segments = [X[ctrl_per_seg*i : ctrl_per_seg*(i+1), :] for i in range(S)]
    Z_traj, _, _ , _ = project_segments_with_coverage(C_segments, A_list, b_list, tol=1e-9, max_as_iters=8, warm_active_seq=True, Ab_prepared=Ab)
    z_traj = Z_traj
    u_traj = np.zeros_like(z_traj)
    z_traj_prev = z_traj.copy()
    rho_list = []

    visualize_voxelgrid_with_path(
        vg,
        path_xyz=init_path,
        plotPath=True,
        tube_radius=0.25,
        grid_opacity=0.25,
        grid_color="blue",
        turns_idx=None,
        mark_start_end=True,
        mark_turns=False,
        A_list=A_list,
        b_list=b_list,
        show_decomposition=True,
        decomp_color="yellow",
        decomp_opacity=0.18,
        smooth_shading=True,
        filename="initial_trajectory"
    )

    exit()

    agg = {
        "sum_step1": 0.0,
        "sum_step2": 0.0,
        "sum_step3": 0.0,
        "sum_iter_total": 0.0,
        "sum_proj_costs_only": 0.0,
        "sum_proj_dp": 0.0,
        "sum_proj_reproj": 0.0,
        "sum_proj_total": 0.0,
        "iters": 0,
    }

    def _write_summary_csv(path, agg, success_flag, meta):
        if agg["iters"] == 0:
            return
        means = {
            "mean_step1_s":           agg["sum_step1"] / agg["iters"],
            "mean_step2_s":           agg["sum_step2"] / agg["iters"],
            "mean_step3_s":           agg["sum_step3"] / agg["iters"],
            "mean_iter_total_s":      agg["sum_iter_total"] / agg["iters"],
            "mean_proj_costs_only_s": agg["sum_proj_costs_only"] / agg["iters"],
            "mean_proj_dp_s":         agg["sum_proj_dp"] / agg["iters"],
            "mean_proj_reproj_s":     agg["sum_proj_reproj"] / agg["iters"],
            "mean_proj_total_s":      agg["sum_proj_total"] / agg["iters"],
        }
        row = {**meta, "iters": agg["iters"], **means, "success": int(bool(success_flag))}
        fieldnames = list(row.keys())
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)


    start_iter = time.perf_counter()
    for k in range(max_iters):
        print(f"--- Iteration {k+1} ---")
        step1_start = time.perf_counter()
        if k == 0:
            z_traj_prev = z_traj.copy()

        ZU = z_traj - u_traj

        if k == 0:
            rho_cache = None
        if (k == 0) or (rho_cache is None) or (abs(rho_cache - rho) > 0):
            LHSx = Hx + rho * (Acx.T @ Acx)
            LHSy = Hy + rho * (Acy.T @ Acy)
            LHSz = Hz + rho * (Acz.T @ Acz)
            Lx_factor = splu(csc_matrix(LHSx))
            Ly_factor = splu(csc_matrix(LHSy))
            Lz_factor = splu(csc_matrix(LHSz))
            rho_cache = rho

        RHSx = rho * (Acx.T @ (ZU[:, 0] - bcx)) - fx
        RHSy = rho * (Acy.T @ (ZU[:, 1] - bcy)) - fy
        RHSz = rho * (Acz.T @ (ZU[:, 2] - bcz)) - fz

        xi_x = Lx_factor.solve(RHSx)
        xi_y = Ly_factor.solve(RHSy)
        xi_z = Lz_factor.solve(RHSz)

        a_x_stacked = coeffs_from_xi_x(xi_x)
        a_y_stacked = coeffs_from_xi_y(xi_y)
        a_z_stacked = coeffs_from_xi_z(xi_z)

        coeffs_x = [Const(a_x_stacked[coeffs_per_seg*i : coeffs_per_seg*(i+1)]) for i in range(S)]
        coeffs_y = [Const(a_y_stacked[coeffs_per_seg*i : coeffs_per_seg*(i+1)]) for i in range(S)]
        coeffs_z = [Const(a_z_stacked[coeffs_per_seg*i : coeffs_per_seg*(i+1)]) for i in range(S)]

        Cx = T_blk @ a_x_stacked
        Cy = T_blk @ a_y_stacked
        Cz = T_blk @ a_z_stacked
        X  = np.column_stack([Cx, Cy, Cz])

        z_traj_prev = z_traj.copy()
        step1_end = time.perf_counter()
        step2_start = time.perf_counter()
        X_plus_u = X + u_traj

        C_segments = [X_plus_u[ctrl_per_seg*i : ctrl_per_seg*(i+1), :] for i in range(S)]  # each (ctrl_per_seg,3)
        Z_traj, _ , _ , proj_timings = project_segments_with_coverage(C_segments, A_list, b_list, tol=1e-9, max_as_iters=8, warm_active_seq=True, Ab_prepared=Ab)
        z_traj = Z_traj
        
        step2_end = time.perf_counter()
        step3_start = time.perf_counter()
        r_inf, s_inf, scale_pri, _ = admm_residuals_cp_3d_from_C(
            Cx, Cy, Cz, z_traj, z_traj_prev, Acx, Acy, Acz, rho
        )

        u_traj = u_traj + (X - z_traj)

        if k >= 3 and (k % 5 == 0):
            rho_new = update_rho_osqp_with_s_cp(
                rho, r_inf, s_inf,
                rho_min=1e-6, rho_max=1e6, step_limit=5.0
            )
            if rho_new != rho:
                scale = rho / rho_new
                u_traj = scale * u_traj
                rho = rho_new
                rho_cache = None

        ATy_x = Acx.T @ (rho * u_traj[:, 0])
        ATy_y = Acy.T @ (rho * u_traj[:, 1])
        ATy_z = Acz.T @ (rho * u_traj[:, 2])
        scale_dual = max(np.linalg.norm(ATy_x, np.inf),
                        np.linalg.norm(ATy_y, np.inf),
                        np.linalg.norm(ATy_z, np.inf))


        rho_list.append(rho)
        step3_end = time.perf_counter()

        eps_pri  = eps_abs_pri  + eps_rel * scale_pri
        eps_dual = eps_abs_dual + eps_rel * max(scale_dual, 1.0)    

        max_diff = float(np.max(np.abs(X - z_traj)))
        iter_total = step3_end - step1_start

        row = {
            "iter": k + 1,
            "step1_primal_s": step1_end - step1_start,
            "step2_projection_s": step2_end - step2_start,
            "step3_dual_residual_s": step3_end - step3_start,
            "iter_total_s": iter_total,
            "proj_costs_only_s": proj_timings.get("proj_costs_only", float("nan")),
            "proj_dp_s": proj_timings.get("proj_dp", float("nan")),
            "proj_reproj_s": proj_timings.get("proj_reproj", float("nan")),
            "proj_total_s": proj_timings.get("proj_total", float("nan")),
            "rho": rho,
            "r_inf": r_inf,
            "s_inf": s_inf,
            "eps_pri": eps_pri,
            "eps_dual": eps_dual,
            "max_abs_XminusZ": max_diff,
        }
        runtime_rows.append(row)

        agg["sum_step1"] += row["step1_primal_s"]
        agg["sum_step2"] += row["step2_projection_s"]
        agg["sum_step3"] += row["step3_dual_residual_s"]
        agg["sum_iter_total"] += row["iter_total_s"]
        agg["sum_proj_costs_only"] += row["proj_costs_only_s"]
        agg["sum_proj_dp"] += row["proj_dp_s"]
        agg["sum_proj_reproj"] += row["proj_reproj_s"]
        agg["sum_proj_total"] += row["proj_total_s"]
        agg["iters"] += 1

        if (r_inf <= eps_pri) and (s_inf <= eps_dual):
            print("Konvergenz erreicht.")
            end_iter = time.perf_counter()
            print(f"Fertig nach {k+1} Iterationen in {end_iter - start_iter:.5f} Sekunden.")

            x_traj = []
            for i in range(S):
                dt = segment_times[i+1] - segment_times[i]
                t_vals = np.linspace(0, dt, m_per_seg)
                ax_i = coeffs_x[i].value
                ay_i = coeffs_y[i].value
                az_i = coeffs_z[i].value
                xs = [evaluate_polynomial(ax_i, t) for t in t_vals]
                ys = [evaluate_polynomial(ay_i, t) for t in t_vals]
                zs = [evaluate_polynomial(az_i, t) for t in t_vals]
                x_traj.append(np.column_stack((xs, ys, zs)))

            final_pts = np.vstack(x_traj)
            curviness = np.sum(np.linalg.norm(np.diff(final_pts, axis=0), axis=1)) / np.linalg.norm(final_pts[-1] - final_pts[0])

            summary_meta.update({
                "curviness_ratio": float(curviness),
            })
            
            _write_summary_csv(runtime_summary_csv, agg, success_flag=True, meta=summary_meta)
        

            visualize_voxelgrid_with_path(
                vg,
                path_xyz=final_pts,
                plotPath=True,
                tube_radius=0.25,
                grid_opacity=0.25,
                grid_color="blue",
                turns_idx=None,
                mark_start_end=True,
                mark_turns=False,
                A_list=A_list,
                b_list=b_list,
                show_decomposition=False,
                decomp_color="yellow",
                decomp_opacity=0.18,
                smooth_shading=True,
                filename="only_trajectory"
            )
            break

    z_traj_prev = z_traj.copy()


