import numpy as np

# ---------- 3D: projection of points onto convex polyhedron Ax - b <= 0 ----------
def project_points_to_polyhedron_pocs(P, A, b, tol=1e-9, max_iters=64):
    """
    Project each row of P (N,3) onto the convex polyhedron {x | A x - b <= 0}
    using POCS (cyclic projections onto half-spaces).

    Returns:
        P_proj: (N,3) projected points
        d2:     (N,)  squared distances ||P_proj - P||^2
    """
    P = np.asarray(P, float).reshape(-1, 3)
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)
    if A.ndim != 2 or A.shape[1] != 3 or b.shape[0] != A.shape[0]:
        raise ValueError(f"A must be (m,3), b (m,), got {A.shape}, {b.shape}")

    # Remove inactive/zero rows
    active = ~np.all(np.isclose(A, 0.0, atol=1e-12), axis=1)
    A = A[active]; b = b[active]
    if A.shape[0] == 0:
        # No constraints -> identity projection
        return P.copy(), np.zeros(P.shape[0])

    # Precompute squared norms of rows
    An2 = np.einsum('ij,ij->i', A, A) + 1e-18

    X = P.copy()
    for _ in range(max_iters):
        # Violations v = A x - b  (N, m)
        v = X @ A.T - b[None, :]
        # If all satisfied within tol, stop
        max_violation = np.max(v)
        if max_violation <= tol:
            break
        # Pick most-violated plane per point (argmax over m)
        j = np.argmax(v, axis=1)           # (N,)
        # Only update the ones actually violating
        mask = v[np.arange(v.shape[0]), j] > tol
        if not np.any(mask):
            break
        jj = j[mask]                        # indices of planes
        # Projection of x onto a_j^T x = b_j:
        # x <- x - ((a·x - b)/||a||^2) a
        a_sel = A[jj]                       # (Nv,3)
        v_sel = (X[mask] * a_sel).sum(axis=1) - b[jj]  # (Nv,)
        X[mask] = X[mask] - (v_sel / An2[jj])[:, None] * a_sel

    d2 = np.sum((X - P) ** 2, axis=1)
    return X, d2


# ---------- 3D: project segments with coverage (DP over contiguous blocks) ----------
def project_segments_with_coverage(C_segments, A_list, b_list):
    """
    3D version.
    C_segments: list of (ctrl_per_seg, 3) arrays (control points per segment, in 3D)
    A_list, b_list: per-region halfspaces with convention A x - b <= 0
    Returns:
        Z_traj: (S*ctrl_per_seg, 3) stacked projected control points
        assign: (S,) region index per segment (contiguous blocks, DP)
        costs:  (S, R) cost matrix (sum of squared distances per segment & region)
    """
    import numpy as np

    # Normalize/validate inputs
    C_segments = [np.asarray(C, float).reshape(-1, 3) for C in C_segments]
    S = len(C_segments)
    R = len(A_list)
    if S < R:
        raise ValueError(f"Erfordert num_segments >= num_regions, aber S={S} < R={R}. "
                         f"Erhöhe num_segments oder fusioniere Regionen.")
    if R == 0:
        raise ValueError("A_list/b_list leer.")

    # Precompute projections & costs for each (segment i, region j)
    costs = np.zeros((S, R), dtype=float)
    projs = {}  # (i,j) -> (ctrl_per_seg, 3)

    # Clean A/b per region once (drop zero rows)
    Ab_clean = []
    for A, b in zip(A_list, b_list):
        A = np.asarray(A, float); b = np.asarray(b, float).reshape(-1)
        active = ~np.all(np.isclose(A, 0.0, atol=1e-12), axis=1)
        A = A[active]; b = b[active]
        Ab_clean.append((A, b))

    for i, C in enumerate(C_segments):
        for j, (A, b) in enumerate(Ab_clean):
            if A.shape[0] == 0:
                # No constraints: identity projection
                P = C.copy(); c = 0.0
            else:
                P, d2 = project_points_to_polyhedron_pocs(C, A, b, tol=1e-9, max_iters=64)
                c = float(np.sum(d2))
            projs[(i, j)] = P
            costs[i, j] = c

    # ---------- DP over contiguous blocks (same as your 2D code) ----------
    pref_cols = [np.concatenate(([0.0], np.cumsum(costs[:, j], axis=0))) for j in range(R)]
    INF = 1e18
    dp_prev = np.full(S + 1, INF); dp_prev[0] = 0.0
    prev_all = []

    for j in range(1, R + 1):
        dp_curr = np.full(S + 1, INF)
        prev_idx = np.full(S + 1, -1, dtype=int)

        pref = pref_cols[j - 1]
        e_min = j
        e_max = S - (R - j)

        s0 = j - 1
        min_so_far = dp_prev[s0] - pref[s0]
        argmin_s   = s0

        for e in range(e_min, e_max + 1):
            val = min_so_far + pref[e]
            dp_curr[e] = val
            prev_idx[e] = argmin_s

            s_new = e - 1
            cand = dp_prev[s_new] - pref[s_new]
            if cand < min_so_far:
                min_so_far = cand
                argmin_s   = s_new

        prev_all.append(prev_idx)
        dp_prev = dp_curr

    ends = []
    e = S
    for j in range(R, 0, -1):
        prev_idx = prev_all[j - 1]
        s = int(prev_idx[e])
        if s < 0:
            raise RuntimeError(f"DP backtrack failed at j={j}, e={e}.")
        ends.append(e)
        e = s
    ends = ends[::-1]
    boundaries = [0] + ends

    bnd = np.array(boundaries, dtype=int)
    if bnd[0] != 0 or bnd[-1] != S:
        raise RuntimeError(f"Ungültige boundaries (Start/Ende): {boundaries} vs S={S}")
    if np.any(np.diff(bnd) <= 0):
        raise RuntimeError(f"Boundaries nicht streng ansteigend: {boundaries}")

    assign = np.full(S, -1, dtype=int)
    for j in range(R):
        s, e = int(bnd[j]), int(bnd[j + 1])
        assign[s:e] = j
    if (assign < 0).any():
        holes = np.where(assign < 0)[0].tolist()
        raise RuntimeError(f"Assignment unvollständig, Lücken in Segmenten {holes}. "
                           f"Boundaries: {boundaries}, S={S}, R={R}")

    # Stack projected control points
    ctrl_per_seg = C_segments[0].shape[0]
    Z_blocks = []
    for i in range(S):
        j = int(assign[i])
        Z_blocks.append(projs[(i, j)])
    Z_traj = np.vstack(Z_blocks).reshape(S * ctrl_per_seg, 3)

    return Z_traj, assign, costs
