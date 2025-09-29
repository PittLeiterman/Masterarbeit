import numpy as np
import time

# ------------------------------------------------------------
# 3D Active-Set-Projektion: exakte Projektion auf Ax <= b
# Löst min_x 1/2||x-p||^2 s.t. A x <= b
# In 3D ist die aktive Menge höchstens Größe 3.
# ------------------------------------------------------------
def _proj_point_qp3d(p, A, b, tol=1e-9, max_as_iters=8, warm_active=None):
    """
    Single-point projection (3D) onto polyhedron {x | A x <= b} using a tiny Active-Set QP.
    Returns:
        x  : (3,) projected point
        d2 : squared distance ||x - p||^2
        I  : tuple of active indices
    """
    p = np.asarray(p, float).reshape(3)
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)
    m = A.shape[0]

    # Triviale Fälle
    if m == 0:
        return p.copy(), 0.0, ()

    # Start: prüfe, ob p schon zulässig ist
    v = A @ p - b
    i_max = int(np.argmax(v)) if m else 0
    if m == 0 or v[i_max] <= tol:
        return p.copy(), 0.0, ()

    # Aktive Menge I initialisieren (Warm-start erlaubt)
    if warm_active is not None and len(warm_active) > 0:
        I = list(warm_active[:3])  # maximal 3
    else:
        I = [i_max]

    # Active-Set Iterationen (sehr klein in der Praxis)
    for _ in range(max_as_iters):
        k = len(I)
        # G = A_I A_I^T (k×k), rhs = A_I p - b_I
        A_I = A[I, :]        # (k,3)
        rhs = A_I @ p - b[I] # (k,)

        # Löse G λ = rhs; G ist klein (1..3)
        G = A_I @ A_I.T
        # numerische Stabilität: winzige Diagonalerhebung
        G.flat[::k+1] += 1e-15

        try:
            # Für k<=3 ist Solve billig; Cholesky falls SPD, sonst Fallback
            if k == 1:
                lam = rhs / G[0, 0]
            else:
                lam = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            # Robustheits-Fallback
            lam = np.linalg.lstsq(G, rhs, rcond=None)[0]

        # KKT-Zeichenbedingung: λ >= 0
        if np.any(lam < -1e-12):
            # Entferne den am meisten negativen Multiplikator (klassisches AS-QP)
            drop = int(np.argmin(lam))
            del I[drop]
            if len(I) == 0:
                # leer -> wähle stärkste Verletzung neu
                i_max = int(np.argmax(A @ p - b))
                if (A @ p - b)[i_max] > tol:
                    I = [i_max]
                else:
                    return p.copy(), 0.0, ()
            continue

        # Kandidatenpunkt
        x = p - A_I.T @ lam  # x = p - A_I^T λ

        # Globale Zulässigkeit prüfen
        v = A @ x - b
        i_max = int(np.argmax(v))
        if v[i_max] <= tol:
            d2 = float(np.dot(x - p, x - p))
            return x, d2, tuple(I)

        # Sonst: neue verletzte Nebenbedingung aufnehmen
        if i_max not in I:
            if len(I) < 3:
                I = I + [i_max]
            else:
                # Wenn bereits 3 aktiv sind: ersetze die schwächste aktive
                # Heuristik: ersetze die mit kleinster λ (am "unwichtigsten")
                if len(lam) > 0:
                    j_rep = int(np.argmin(lam))
                    I[j_rep] = i_max
                else:
                    # degenerater Fall
                    I[-1] = i_max
        else:
            # Sollte selten passieren; breche ab, um Endlosschleifen zu vermeiden
            break

    # Fallback: gib letzten x zurück (approx)
    x = p if 'x' not in locals() else x
    d2 = float(np.dot(x - p, x - p))
    return x, d2, tuple(I)


def project_points_to_polyhedron_qp3d(P, A, b, tol=1e-9, max_as_iters=8, warm_active_seq=False):
    """
    Vectorized wrapper über Punkte (N,3).
    warm_active_seq=True: aktives Set entlang der Punktfolge propagieren (gut bei Trajektorien).
    Returns:
        P_proj: (N,3)
        d2    : (N,)
        active_sets: list of tuples (für Debug/Warm-start Downstream)
    """
    P = np.asarray(P, float).reshape(-1, 3)
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)

    # Zero-rows entfernen (wie bei dir)
    active = ~np.all(np.isclose(A, 0.0, atol=1e-12), axis=1)
    A = A[active]; b = b[active]
    if A.shape[0] == 0:
        return P.copy(), np.zeros(P.shape[0]), [() for _ in range(P.shape[0])]

    N = P.shape[0]
    X = np.empty_like(P)
    d2 = np.empty(N, dtype=float)
    active_sets = [()]*N

    warm = None
    for i in range(N):
        x, di2, I = _proj_point_qp3d(P[i], A, b, tol=tol, max_as_iters=max_as_iters, warm_active=warm)
        X[i] = x
        d2[i] = di2
        active_sets[i] = I
        if warm_active_seq:
            warm = I
    return X, d2, active_sets


# ------------------------------------------------------------
# 3D: project segments with coverage (DP) – Speicher/Speed-optimiert
# 1) Nur Kosten berechnen (streaming, ohne Proj-Puffer).
# 2) DP lösen.
# 3) Nur zugewiesene Projektionen neu berechnen und stapeln.
# ------------------------------------------------------------
def project_segments_with_coverage(C_segments, A_list, b_list, *,
                                   tol=1e-9, max_as_iters=8,
                                   warm_active_seq=True, verbose_timing=True):
    """
    C_segments: list of (ctrl_per_seg,3)
    A_list/b_list: pro Region
    Returns:
        Z_traj: (S*ctrl_per_seg, 3)
        assign: (S,)
        costs:  (S, R)  (nur Summen der d²)
    """
    start_total = time.perf_counter()

    # Normalize
    C_segments = [np.asarray(C, float).reshape(-1, 3) for C in C_segments]
    S = len(C_segments)
    R = len(A_list)
    if S < R:
        raise ValueError(f"Erfordert num_segments >= num_regions, aber S={S} < R={R}. "
                         f"Erhöhe num_segments oder fusioniere Regionen.")
    if R == 0:
        raise ValueError("A_list/b_list leer.")
    ctrl_per_seg = C_segments[0].shape[0]

    # Clean A/b pro Region
    Ab = []
    for A, b in zip(A_list, b_list):
        A = np.asarray(A, float); b = np.asarray(b, float).reshape(-1)
        active = ~np.all(np.isclose(A, 0.0, atol=1e-12), axis=1)
        A = A[active]; b = b[active]
        Ab.append((A, b))

    # ---- 1) Kosten berechnen (ohne Projektionen puffern)
    t0 = time.perf_counter()
    costs = np.zeros((S, R), dtype=float)
    for j, (A, b) in enumerate(Ab):
        warm_active = None
        for i, C in enumerate(C_segments):
            if A.shape[0] == 0:
                c = 0.0
                warm_active = None
            else:
                # Nur Kosten (Summe d²), optional Warm-start entlang der Segmente
                X, d2, _ = project_points_to_polyhedron_qp3d_numba(
                    C, A, b, tol=tol, max_as_iters=max_as_iters, warm_active_seq=warm_active_seq
                )
                c = float(np.sum(d2))
            costs[i, j] = c
    t1 = time.perf_counter()

    # ---- 2) DP (unverändert, aber ohne projs-Speicher)
    t_dp0 = time.perf_counter()
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
    t_dp1 = time.perf_counter()

    # ---- 3) Nur die tatsächlich zugewiesenen Projektionen neu berechnen
    t2 = time.perf_counter()
    Z_blocks = []
    # Warm-starts pro Region entlang zusammenhängender Blöcke
    for j in range(R):
        idxs = np.where(assign == j)[0]
        if idxs.size == 0:
            continue
        A, b = Ab[j]
        warm_active = None
        # zusammenhängende Teilstücke pro Region finden
        runs = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
        for run in runs:
            warm_active = None
            for i in run:
                C = C_segments[i]
                if A.shape[0] == 0:
                    X = C.copy()
                else:
                    X, d2, _ = project_points_to_polyhedron_qp3d_numba(
                        C, A, b, tol=tol, max_as_iters=max_as_iters, warm_active_seq=warm_active_seq
                    )
                Z_blocks.append(X)

    if len(Z_blocks) != S:
        # Falls Reihenfolge durch Blöcke nicht natürlich: sortieren
        # (Normalerweise kommt genau S Stücke in Segment-Reihenfolge.)
        # Backup: in Segmentreihenfolge zusammenbauen
        Z_blocks = []
        warm_region_active = [None]*R
        for i in range(S):
            j = int(assign[i])
            A, b = Ab[j]
            C = C_segments[i]
            if A.shape[0] == 0:
                X = C.copy()
            else:
                X, d2, _ = project_points_to_polyhedron_qp3d_numba(
                    C, A, b, tol=tol, max_as_iters=max_as_iters, warm_active_seq=warm_active_seq
                )
            Z_blocks.append(X)

    Z_traj = np.vstack(Z_blocks).reshape(S * ctrl_per_seg, 3)
    t3 = time.perf_counter()

    if verbose_timing:
        print(f"Costs-only pass: {t1 - t0:.6f}s | DP: {t_dp1 - t_dp0:.6f}s | Re-proj assigned: {t3 - t2:.6f}s | Total: {time.perf_counter() - start_total:.6f}s")

    return Z_traj, assign, costs


from numba import njit, prange
import numpy as np

@njit(cache=True, fastmath=True)
def _dot3(a0,a1,a2, x0,x1,x2):
    return a0*x0 + a1*x1 + a2*x2

@njit(cache=True, fastmath=True)
def _solve1(G00, r0):
    return r0 / G00

@njit(cache=True, fastmath=True)
def _solve2(G00,G01,G11, r0,r1):
    # [[G00, G01],[G01, G11]] [l0,l1]=[r0,r1]
    det = G00*G11 - G01*G01
    if abs(det) < 1e-18:
        # least-squares fallback
        det = 1e-18
    inv00 =  G11/det
    inv01 = -G01/det
    inv11 =  G00/det
    l0 = inv00*r0 + inv01*r1
    l1 = inv01*r0 + inv11*r1
    return l0, l1

@njit(cache=True, fastmath=True)
def _solve3(G, r):
    # 3x3 solve via Gaussian elimination (tiny, stable enough here)
    A = np.empty((3,4))
    A[0,0]=G[0,0]; A[0,1]=G[0,1]; A[0,2]=G[0,2]; A[0,3]=r[0]
    A[1,0]=G[1,0]; A[1,1]=G[1,1]; A[1,2]=G[1,2]; A[1,3]=r[1]
    A[2,0]=G[2,0]; A[2,1]=G[2,1]; A[2,2]=G[2,2]; A[2,3]=r[2]
    # Pivot 0
    if abs(A[0,0]) < 1e-18: A[0,0] = 1e-18
    f = 1.0/A[0,0]
    for j in range(1,4): A[0,j] *= f
    A[0,0] = 1.0
    # Eliminate col 0
    for i in range(1,3):
        m = A[i,0]
        for j in range(1,4):
            A[i,j] -= m*A[0,j]
        A[i,0]=0.0
    # Pivot 1
    if abs(A[1,1]) < 1e-18: A[1,1] = 1e-18
    f = 1.0/A[1,1]
    for j in range(2,4): A[1,j] *= f
    A[1,1] = 1.0
    # Eliminate col 1
    m = A[0,1]
    for j in range(2,4): A[0,j] -= m*A[1,j]
    A[0,1]=0.0
    m = A[2,1]
    for j in range(2,4): A[2,j] -= m*A[1,j]
    A[2,1]=0.0
    # Pivot 2
    if abs(A[2,2]) < 1e-18: A[2,2] = 1e-18
    f = 1.0/A[2,2]
    A[2,3] *= f
    A[2,2] = 1.0
    # Back-substitute
    A[1,3] -= A[1,2]*A[2,3]; A[1,2]=0.0
    A[0,3] -= A[0,2]*A[2,3]; A[0,2]=0.0
    A[0,3] -= A[0,2]*A[2,3]
    return A[0,3], A[1,3], A[2,3]

@njit(cache=True, fastmath=True)
def qp3d_point_njit(p, A, b, tol=1e-9, max_as_iters=8, warm0=-1, warm1=-1, warm2=-1):
    """
    Active-Set QP für einen Punkt.
    warm0..2: initiale aktive Indizes (oder -1).
    Rückgabe: x(3,), d2, I0,I1,I2 (=-1 wenn inaktiv)
    """
    m = A.shape[0]
    x0, x1, x2 = p[0], p[1], p[2]

    if m == 0:
        return x0, x1, x2, 0.0, -1, -1, -1

    # Prüfe Feasibility von p
    vmax = -1e30; imax = -1
    for i in range(m):
        v = _dot3(A[i,0],A[i,1],A[i,2], x0,x1,x2) - b[i]
        if v > vmax:
            vmax = v; imax = i
    if vmax <= tol:
        return x0, x1, x2, 0.0, -1, -1, -1

    # Aktives Set (max 3)
    I0, I1, I2 = -1, -1, -1
    if warm0 >= 0: I0 = warm0
    if warm1 >= 0:
        if I0 == -1: I0 = warm1
        else: I1 = warm1
    if warm2 >= 0:
        if I0 == -1: I0 = warm2
        elif I1 == -1: I1 = warm2
        else: I2 = warm2
    if I0 == -1:
        I0 = imax

    for _ in range(max_as_iters):
        # baue A_I und rhs
        k = 0
        idx = np.empty(3, dtype=np.int64)
        if I0 >= 0: idx[k]=I0; k+=1
        if I1 >= 0: idx[k]=I1; k+=1
        if I2 >= 0: idx[k]=I2; k+=1

        # rhs = A_I p - b_I ; G = A_I A_I^T
        if k == 1:
            i0 = idx[0]
            a00,a01,a02 = A[i0,0],A[i0,1],A[i0,2]
            r0 = a00*p[0] + a01*p[1] + a02*p[2] - b[i0]
            G00 = a00*a00 + a01*a01 + a02*a02 + 1e-15
            lam0 = _solve1(G00, r0)
            if lam0 < -1e-12:
                # Drop negatives
                I0, I1, I2 = -1, I1, I2
                continue
            x0 = p[0] - (a00*lam0)
            x1 = p[1] - (a01*lam0)
            x2 = p[2] - (a02*lam0)
        elif k == 2:
            i0, i1 = idx[0], idx[1]
            a00,a01,a02 = A[i0,0],A[i0,1],A[i0,2]
            b00 = b[i0]
            a10,a11,a12 = A[i1,0],A[i1,1],A[i1,2]
            b10 = b[i1]
            r0 = a00*p[0] + a01*p[1] + a02*p[2] - b00
            r1 = a10*p[0] + a11*p[1] + a12*p[2] - b10
            G00 = a00*a00 + a01*a01 + a02*a02 + 1e-15
            G01 = a00*a10 + a01*a11 + a02*a12
            G11 = a10*a10 + a11*a11 + a12*a12 + 1e-15
            l0,l1 = _solve2(G00,G01,G11, r0,r1)
            if l0 < -1e-12 or l1 < -1e-12:
                # Drop most negative
                if l0 <= l1:
                    I0 = I1; I1 = I2; I2 = -1
                else:
                    I1 = I2; I2 = -1
                continue
            x0 = p[0] - (a00*l0 + a10*l1)
            x1 = p[1] - (a01*l0 + a11*l1)
            x2 = p[2] - (a02*l0 + a12*l1)
        else:
            # k == 3
            i0,i1,i2 = idx[0], idx[1], idx[2]
            AIt = np.empty((3,3))
            AIt[0,0]=A[i0,0]; AIt[0,1]=A[i0,1]; AIt[0,2]=A[i0,2]
            AIt[1,0]=A[i1,0]; AIt[1,1]=A[i1,1]; AIt[1,2]=A[i1,2]
            AIt[2,0]=A[i2,0]; AIt[2,1]=A[i2,1]; AIt[2,2]=A[i2,2]
            rhs = np.empty(3)
            rhs[0] = AIt[0,0]*p[0] + AIt[0,1]*p[1] + AIt[0,2]*p[2] - b[i0]
            rhs[1] = AIt[1,0]*p[0] + AIt[1,1]*p[1] + AIt[1,2]*p[2] - b[i1]
            rhs[2] = AIt[2,0]*p[0] + AIt[2,1]*p[1] + AIt[2,2]*p[2] - b[i2]
            G = np.empty((3,3))
            # G = A_I A_I^T
            for r in range(3):
                for c in range(3):
                    G[r,c] = (AIt[r,0]*AIt[c,0] + AIt[r,1]*AIt[c,1] + AIt[r,2]*AIt[c,2])
            G[0,0]+=1e-15; G[1,1]+=1e-15; G[2,2]+=1e-15
            l0,l1,l2 = _solve3(G, rhs)
            if l0 < -1e-12 or l1 < -1e-12 or l2 < -1e-12:
                # Drop most negative
                mn = l0; pos = 0
                if l1 < mn: mn=l1; pos=1
                if l2 < mn: mn=l2; pos=2
                if pos == 0: I0 = I1; I1 = I2; I2 = -1
                elif pos == 1: I1 = I2; I2 = -1
                else: I2 = -1
                continue
            x0 = p[0] - (AIt[0,0]*l0 + AIt[1,0]*l1 + AIt[2,0]*l2)
            x1 = p[1] - (AIt[0,1]*l0 + AIt[1,1]*l1 + AIt[2,1]*l2)
            x2 = p[2] - (AIt[0,2]*l0 + AIt[1,2]*l1 + AIt[2,2]*l2)

        # Check globale Feasibility
        vmax = -1e30; imax = -1
        for i in range(m):
            v = _dot3(A[i,0],A[i,1],A[i,2], x0,x1,x2) - b[i]
            if v > vmax:
                vmax = v; imax = i
        if vmax <= tol:
            dx0 = x0 - p[0]; dx1 = x1 - p[1]; dx2 = x2 - p[2]
            d2 = dx0*dx0 + dx1*dx1 + dx2*dx2
            # Rückgabe inkl. aktives Set
            # Re-konstruiere I0,I1,I2 aus der lokalen Reihenfolge:
            k = 0
            i0=-1; i1=-1; i2=-1
            if I0 >= 0: 
                i0 = I0; k+=1
            if I1 >= 0: 
                if k==0: i0 = I1
                elif k==1: i1 = I1
                k+=1
            if I2 >= 0:
                if k==0: i0=I2
                elif k==1: i1=I2
                else: i2=I2
            return x0, x1, x2, d2, i0,i1,i2

        # Neue verletzte Bedingung aufnehmen/ersetzen
        if I0 != imax and I1 != imax and I2 != imax:
            if I0 == -1: I0 = imax
            elif I1 == -1: I1 = imax
            elif I2 == -1: I2 = imax
            else:
                # Heuristik: ersetze I2
                I2 = imax
        else:
            # Sicherheitsabbruch
            break

    dx0 = x0 - p[0]; dx1 = x1 - p[1]; dx2 = x2 - p[2]
    d2 = dx0*dx0 + dx1*dx1 + dx2*dx2
    return x0, x1, x2, d2, I0,I1,I2

@njit(cache=True, fastmath=True)
def project_points_to_polyhedron_qp3d_numba(P, A, b, tol=1e-9, max_as_iters=8, warm_active_seq=True):
    """
    Numba-batch Wrapper. Gibt (X, d2, active_sets) zurück.
    active_sets: (N,3) mit -1 für inaktive Plätze.
    """
    N = P.shape[0]
    X = np.empty_like(P)
    d2 = np.empty(N)
    active = np.empty((N,3), dtype=np.int64)

    warm0=-1; warm1=-1; warm2=-1
    for i in range(N):
        x0,x1,x2,di2,I0,I1,I2 = qp3d_point_njit(P[i], A, b, tol, max_as_iters,
                                                 warm0,warm1,warm2 if warm_active_seq else -1)
        X[i,0]=x0; X[i,1]=x1; X[i,2]=x2
        d2[i]=di2
        active[i,0]=I0; active[i,1]=I1; active[i,2]=I2
        if warm_active_seq:
            warm0, warm1, warm2 = I0, I1, I2
    return X, d2, active
