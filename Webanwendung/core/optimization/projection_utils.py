import numpy as np

def project_point_to_polyhedron(A, b, point, tol=1e-10):
    """
    Robuster 2D-Punkt-Input und Projektion in ein konvexes Polyeder {x | A x <= b}.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)

    # --- robustes Punkt-Shape ---
    p = np.asarray(point, float).reshape(-1)
    if p.size != 2:
        raise ValueError(f"Expected 2D point, got shape {np.asarray(point).shape} (size {p.size})")
    p = p[:2]

    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("2D only (A must be m x 2).")

    # Bereits zulässig?
    if np.all(A @ p <= b + tol):
        return p.copy()

    m = A.shape[0]
    cand = []

    # (1) Orthogonale Projektion auf jede Halbraum-Grenze a_i^T x = b_i, dann Feasibility check
    if m > 0:
        An2 = np.einsum("ij,ij->i", A, A)
        denom_ok = An2 > tol
        if np.any(denom_ok):
            t = (A @ p - b) / np.maximum(An2, tol)
            X = p - t[:, None] * A
            feas = np.all(A @ X.T <= b[:, None] + tol, axis=0)
            cand.append(X[denom_ok & feas])

    # (2) Schnittpunkte aller Paare (Eckpunkte), dann Feasibility
    if m >= 2:
        I, J = np.triu_indices(m, 1)
        ai, aj = A[I], A[J]
        bi, bj = b[I], b[J]
        det = ai[:, 0]*aj[:, 1] - ai[:, 1]*aj[:, 0]
        mask = np.abs(det) > tol
        if np.any(mask):
            ai, aj, bi, bj, det = ai[mask], aj[mask], bi[mask], bj[mask], det[mask]
            Xv = np.empty((det.shape[0], 2))
            Xv[:, 0] = (aj[:, 1]*bi - ai[:, 1]*bj) / det
            Xv[:, 1] = (-aj[:, 0]*bi + ai[:, 0]*bj) / det
            feas = np.all(A @ Xv.T <= b[:, None] + tol, axis=0)
            cand.append(Xv[feas])

    # Fallbacks
    if not cand:
        return p.copy()
    C = np.vstack([c for c in cand if c.size]) if len(cand) > 1 else cand[0]
    if C.size == 0:
        return p.copy()

    # Nächster zulässiger Punkt
    d2 = np.sum((C - p) ** 2, axis=1)
    return C[np.argmin(d2)]


def project_segments_with_coverage(C_segments, A_list, b_list):
    import numpy as np

    def _proj_cost_for_region(C_seg, A, b):
        P = np.vstack([project_point_to_polyhedron(A, b, p) for p in C_seg])
        cost = float(np.sum((P - C_seg)**2))
        return P, cost

    C_segments = [np.asarray(C, float).reshape(-1, 2) for C in C_segments]
    S = len(C_segments)
    R = len(A_list)
    if S < R:
        raise ValueError(f"Erfordert num_segments >= num_regions, aber S={S} < R={R}. "
                         f"Erhöhe num_segments oder fusioniere Regionen.")

    # --- Precompute Projektionen & Kosten ---
    costs = np.zeros((S, R), dtype=float)
    projs = {}  # (i,j) -> (ctrl_per_seg,2)
    for i, C in enumerate(C_segments):
        for j, (A, b) in enumerate(zip(A_list, b_list)):
            P, c = _proj_cost_for_region(C, A, b)
            projs[(i, j)] = P
            costs[i, j]  = c

    # --- DP über zusammenhängende Blöcke ---
    INF = 1e18
    prefix = np.vstack([np.zeros((1, R)), np.cumsum(costs, axis=0)])  # (S+1,R)

    dp   = np.full((R + 1, S + 1), INF, dtype=float)
    prev = np.full((R + 1, S + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for j in range(1, R + 1):
        e_min = j
        e_max = S - (R - j)
        for e in range(e_min, e_max + 1):
            best = INF
            best_s = -1
            col = j - 1
            s_min = j - 1
            s_max = e - 1
            for s in range(s_min, s_max + 1):
                block_cost = prefix[e, col] - prefix[s, col]  # Sum costs[s:e, j-1]
                val = dp[j - 1, s] + block_cost
                if val < best:
                    best = val
                    best_s = s
            dp[j, e] = best
            prev[j, e] = best_s

    # --- Rekonstruktion ---
    # --- Rekonstruktion: Blockenden statt Blockstarts verwenden ---
    ends = [S]          # e_R
    e = S
    for j in range(R, 1, -1):         # bis j=2 zurücklaufen; s_1=0 wollen wir NICHT aufnehmen
        s = int(prev[j, e])
        if s < 0:
            raise RuntimeError(
                f"DP backtrack failed at j={j}, e={e}. "
                f"dp[j, e]={dp[j, e]}, prev[j, e]={prev[j, e]}."
            )
        ends.append(s)                # e_{j-1} = s_j
        e = s

    ends = ends[::-1]                 # [e_1, e_2, ..., e_{R-1}, S]
    boundaries = [0] + ends           # [0, e_1, e_2, ..., e_{R-1}, S]


    # --- Sanity Checks für Grenzen ---
    bnd = np.array(boundaries, dtype=int)
    if bnd[0] != 0 or bnd[-1] != S:
        raise RuntimeError(f"Ungültige boundaries (Start/Ende): {boundaries} vs S={S}")
    if np.any(np.diff(bnd) <= 0):
        raise RuntimeError(f"Boundaries nicht streng ansteigend: {boundaries}")

    # --- Assign bauen, initial mit -1 ---
    assign = np.full(S, -1, dtype=int)
    for j in range(R):
        s, e = int(bnd[j]), int(bnd[j + 1])
        assign[s:e] = j

    if (assign < 0).any():
        holes = np.where(assign < 0)[0].tolist()
        raise RuntimeError(f"Assignment unvollständig, Lücken in Segmenten {holes}. "
                           f"Boundaries: {boundaries}, S={S}, R={R}")

    # --- Z_traj zusammensetzen ---
    Z_blocks = []
    ctrl_per_seg = C_segments[0].shape[0]
    for i in range(S):
        j = int(assign[i])
        if (i, j) not in projs:
            raise KeyError(f"Projektion fehlt für (seg={i}, region={j}). "
                           f"S={S}, R={R}, assign[{i}]={j}")
        Z_blocks.append(projs[(i, j)])
    Z_traj = np.vstack(Z_blocks).reshape(S * ctrl_per_seg, 2)

    return Z_traj, assign, costs
