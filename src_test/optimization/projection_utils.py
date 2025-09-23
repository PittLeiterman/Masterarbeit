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

    def _proj_cost_for_region_batch(C_seg, poly):
        # C_seg: (ctrl_per_seg,2)
        P, d2 = project_points_to_polygon(C_seg, poly)
        return P, float(np.sum(d2))

    C_segments = [np.asarray(C, float).reshape(-1, 2) for C in C_segments]
    S = len(C_segments)
    R = len(A_list)
    if S < R:
        raise ValueError(f"Erfordert num_segments >= num_regions, aber S={S} < R={R}. "
                         f"Erhöhe num_segments oder fusioniere Regionen.")
    

    polys = [halfspace_to_polygon_2d(A, b) for A, b in zip(A_list, b_list)]

    # --- Precompute Projektionen & Kosten ---
    costs = np.zeros((S, R), dtype=float)
    projs = {}  # (i,j) -> (ctrl_per_seg,2)
    for i, C in enumerate(C_segments):
        for j, poly in enumerate(polys):
            P, c = _proj_cost_for_region_batch(C, poly)
            projs[(i, j)] = P
            costs[i, j]  = c


        # --- DP über zusammenhängende Blöcke in O(S·R) ---
    # costs: (S, R)
    # Für jede Region j bauen wir Prefixsummen über die Segmente:
    pref_cols = [np.concatenate(([0.0], np.cumsum(costs[:, j], axis=0))) for j in range(R)]

    INF = 1e18
    dp_prev = np.full(S + 1, INF)
    dp_prev[0] = 0.0          # dp[0,0] = 0
    prev_all = []             # speichert pro j die Vorgänger-Indizes für Backtracking


    for j in range(1, R + 1):
        dp_curr = np.full(S + 1, INF)
        prev_idx = np.full(S + 1, -1, dtype=int)

        pref = pref_cols[j - 1]

        # gültiger Bereich für e: mindestens j Segmente, höchstens S-(R-j)
        e_min = j
        e_max = S - (R - j)

        # Wir halten min_{s in [j-1, e-1]} (dp_prev[s] - pref[s]) als laufendes Minimum
        # Startfenster enthält s = j-1
        s0 = j - 1
        min_so_far = dp_prev[s0] - pref[s0]
        argmin_s   = s0

        for e in range(e_min, e_max + 1):
            # optimaler Wert für dieses e mit aktuellem min_so_far
            val = min_so_far + pref[e]
            dp_curr[e] = val
            prev_idx[e] = argmin_s

            # Fenster um neuen s = e-1 erweitern (für nächstes e)
            s_new = e - 1
            cand = dp_prev[s_new] - pref[s_new]
            if cand < min_so_far:
                min_so_far = cand
                argmin_s   = s_new

        prev_all.append(prev_idx)
        dp_prev = dp_curr  # nächste Schicht

    # --- Rekonstruktion der Grenzen (boundaries) in O(R) ---
    ends = []        # sammelt [e_R=S, e_{R-1}, ..., e_1]
    e = S
    for j in range(R, 0, -1):
        prev_idx = prev_all[j - 1]
        s = int(prev_idx[e])
        if s < 0:
            raise RuntimeError(f"DP backtrack failed at j={j}, e={e}.")
        ends.append(e)   # e_j
        e = s            # s_j = e_{j-1}

    # Jetzt in aufsteigende Reihenfolge bringen: [e_1, e_2, ..., e_R=S]
    ends = ends[::-1]
    boundaries = [0] + ends


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


def halfspace_to_polygon_2d(A, b, tol=1e-9):
    """
    Grob robuste H->V Konversion in 2D:
    Nimmt alle Paar-Schnittpunkte und behält nur die, die Ax<=b erfüllen,
    danach ConvexHull sortiert sie CCW.
    Einmal pro Region aufrufen!
    """
    import numpy as np
    from scipy.spatial import ConvexHull

    A = np.asarray(A, float); b = np.asarray(b, float).reshape(-1)
    m = A.shape[0]
    if m == 0:
        return np.empty((0,2))

    # Alle Linien-Schnittpunkte
    I, J = np.triu_indices(m, 1)
    ai, aj = A[I], A[J]; bi, bj = b[I], b[J]
    det = ai[:,0]*aj[:,1] - ai[:,1]*aj[:,0]
    mask = np.abs(det) > tol
    if not np.any(mask):
        return np.empty((0,2))
    ai, aj, bi, bj, det = ai[mask], aj[mask], bi[mask], bj[mask], det[mask]
    X = np.empty((det.shape[0], 2))
    X[:,0] = (aj[:,1]*bi - ai[:,1]*bj) / det
    X[:,1] = (-aj[:,0]*bi + ai[:,0]*bj) / det

    # Nur Punkte behalten, die alle Ungleichungen erfüllen
    feas = np.all(A @ X.T <= b[:,None] + tol, axis=0)
    V = X[feas]
    if V.shape[0] < 3:
        return V  # entartet oder Linie/Punkt

    hull = ConvexHull(V)
    poly = V[hull.vertices]  # CCW
    return poly


def project_points_to_polygon(P, poly):
    """
    P: (N,2) Punkte, poly: (K,2) CCW-Polygon
    Rückgabe: (N,2) projizierte Punkte + quadratische Abstände
    """
    import numpy as np
    P = np.asarray(P, float).reshape(-1,2)
    if poly.size == 0:
        return P.copy(), np.zeros(P.shape[0])

    K = poly.shape[0]
    # Kanten
    Q = poly
    Q_next = np.roll(Q, -1, axis=0)
    E = Q_next - Q             # (K,2)
    EL2 = np.sum(E*E, axis=1)  # (K,)

    # Inside-Test: alle (n·(p-q)) <= 0 mit Außen-Normale n=(e_y,-e_x)
    # CCW -> outward normal n = (E[:,1], -E[:,0])
    N = np.column_stack([E[:,1], -E[:,0]])      # (K,2)
    # Für numerische Stabilität normalisieren (optional)
    Nn = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-15)

    # Für alle Punkte: max_k (Nn_k · (P - Q_k)) <= 0  => inside
    # Broadcasting:
    diff = P[:,None,:] - Q[None,:,:]            # (N,K,2)
    signed = np.einsum('nkd,kd->nk', diff, Nn)  # (N,K)
    inside = (np.max(signed, axis=1) <= 1e-12)

    # Default: Ergebnis ist der Punkt selbst (falls inside)
    P_proj = P.copy()
    d2 = np.zeros(P.shape[0])

    # Für outside-Punkte: auf jede Kante projizieren und Minimum nehmen
    idx = np.where(~inside)[0]
    if idx.size:
        P_out = P[idx]                           # (N0,2)
        # Projektion auf Segmente Q->Q_next
        # t = clamp( ((P-Q)·E) / ||E||^2, 0, 1 )
        PQ = P_out[:,None,:] - Q[None,:,:]       # (N0,K,2)
        t = np.einsum('nkd,kd->nk', PQ, E) / (EL2[None,:] + 1e-15)
        t = np.clip(t, 0.0, 1.0)
        X = Q[None,:,:] + t[:,:,None]*E[None,:,:]  # (N0,K,2)
        # d2 zu allen Kanten
        diff2 = P_out[:,None,:] - X
        d2_all = np.sum(diff2*diff2, axis=2)       # (N0,K)
        jmin = np.argmin(d2_all, axis=1)           # (N0,)
        P_proj[idx] = X[np.arange(idx.size), jmin]
        d2[idx] = d2_all[np.arange(idx.size), jmin]

    return P_proj, d2
