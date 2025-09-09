# optimization/minco.py
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix, vstack as sp_vstack, hstack as sp_hstack, block_diag as sp_block_diag
from scipy.sparse.linalg import splu

# ---------- snap cost (same as your current formula) ----------
def snap_Q_quintic(dt: float) -> np.ndarray:
    Q = np.zeros((6, 6))
    Q[4, 4] = 24**2 * dt
    Q[4, 5] = Q[5, 4] = 24*120 * (dt**2) / 2.0
    Q[5, 5] = 120**2 * (dt**3) / 3.0
    return Q

def stack_Q(segment_times):
    S = len(segment_times) - 1
    Q_blocks = []
    for i in range(S):
        dt = segment_times[i+1] - segment_times[i]
        Q_blocks.append(csr_matrix(snap_Q_quintic(dt)))
    return sp_block_diag(Q_blocks, format="csr")  # (6S x 6S)

# ---------- monomial rows ----------
def mono_rows(dt: float):
    T0      = np.array([1, 0, 0, 0, 0, 0])
    T0dot   = np.array([0, 1, 0, 0, 0, 0])
    T0dd    = np.array([0, 0, 2, 0, 0, 0])
    T0ddd   = np.array([0, 0, 0, 6, 0, 0])

    T1      = np.array([1, dt, dt**2, dt**3, dt**4, dt**5])
    T1dot   = np.array([0, 1, 2*dt, 3*dt**2, 4*dt**3, 5*dt**4])
    T1dd    = np.array([0, 0, 2, 6*dt, 12*dt**2, 20*dt**3])
    T1ddd   = np.array([0, 0, 0, 6, 24*dt, 60*dt**2])
    return dict(T0=T0, T0dot=T0dot, T0dd=T0dd, T0ddd=T0ddd,
                T1=T1, T1dot=T1dot, T1dd=T1dd, T1ddd=T1ddd)

# ---------- constraints Aeq a = B p + h ----------
def build_constraints(segment_times, v_start: float, v_end: float):
    """
    Variables are all coefficients 'a' stacked for all segments (6S).
    Constraints:
      - position at every segment start equals p_i
      - position at every segment end   equals p_{i+1}
      - start velocity equals v_start, end velocity equals v_end
      - continuity of vel/acc/jerk across interior knots
    Right-hand side is affine in knot positions p=[p0..pS]; velocities are constants in 'h'.
    Returns:
      Aeq: (m x 6S) CSR
      Bp : (m x (S+1)) CSR   (multiplies knot positions)
      h  : (m,) numpy        (constants from boundary velocities; others zero)
    """
    S = len(segment_times) - 1
    rows_A, rows_B, rhs_h = [], [], []

    def seg_slice(i): return slice(6*i, 6*(i+1))

    # 1) Position at segment starts/ends equals p_i, p_{i+1}
    for i in range(S):
        dt = segment_times[i+1] - segment_times[i]
        R = mono_rows(dt)

        # start position = p_i
        rA = np.zeros(6*S); rA[seg_slice(i)] = R["T0"]
        rB = np.zeros(S+1); rB[i] = 1.0
        rows_A.append(rA); rows_B.append(rB); rhs_h.append(0.0)

        # end position = p_{i+1}
        rA = np.zeros(6*S); rA[seg_slice(i)] = R["T1"]
        rB = np.zeros(S+1); rB[i+1] = 1.0
        rows_A.append(rA); rows_B.append(rB); rhs_h.append(0.0)

    # 2) Start and end velocity
    dt0 = segment_times[1] - segment_times[0]
    R0  = mono_rows(dt0)
    rA = np.zeros(6*S); rA[seg_slice(0)] = R0["T0dot"]
    rows_A.append(rA); rows_B.append(np.zeros(S+1)); rhs_h.append(v_start)

    dte = segment_times[-1] - segment_times[-2]
    Re  = mono_rows(dte)
    rA = np.zeros(6*S); rA[seg_slice(S-1)] = Re["T1dot"]
    rows_A.append(rA); rows_B.append(np.zeros(S+1)); rhs_h.append(v_end)

    # 3) Continuity of vel/acc/jerk across interior knots (i = 0..S-2)
    for i in range(S-1):
        dt = segment_times[i+1] - segment_times[i]
        R  = mono_rows(dt)
        dt2 = segment_times[i+2] - segment_times[i+1]
        Rn  = mono_rows(dt2)

        for key_end, key_start in [("T1dot","T0dot"), ("T1dd","T0dd"), ("T1ddd","T0ddd")]:
            rA = np.zeros(6*S)
            rA[seg_slice(i)]   = R[key_end]
            rA[seg_slice(i+1)] = -Rn[key_start]
            rows_A.append(rA); rows_B.append(np.zeros(S+1)); rhs_h.append(0.0)

    Aeq = csr_matrix(np.vstack(rows_A))
    Bp  = csr_matrix(np.vstack(rows_B))
    h   = np.array(rhs_h, dtype=float)
    return Aeq, Bp, h

# ---------- MINCO mapping precompute ----------
def precompute_mapping(segment_times, p0: float, pS: float, v_start: float, v_end: float):
    """
    Returns a function that maps interior waypoints xi=[p1..p_{S-1}] to the stacked
    coefficients a = M xi + c (shape 6S). Also returns Q_blk for later use.
    """
    S = len(segment_times) - 1
    Q_blk = stack_Q(segment_times) * 2.0             # use H = 2Q

    Aeq, Bp, h = build_constraints(segment_times, v_start, v_end)
    H = Q_blk.tocsc()

    # Build KKT and factor once
    Z = csr_matrix((Aeq.shape[0], Aeq.shape[0]))
    KKT = sp_vstack([sp_hstack([H, Aeq.T]), sp_hstack([Aeq, Z])]).tocsc()
    KKT_lu = splu(KKT)

    # Split p into [p0, xi..., pS]
    # Selector that expands xi -> full p (S+1), placing xi into indices 1..S-1
    data, rows, cols = [], [], []
    for j in range(S-1):
        rows.append(1 + j)   # target row in p
        cols.append(j)       # column in xi
        data.append(1.0)
    S_expand = csr_matrix((data, (rows, cols)), shape=(S+1, S-1))

    p_const = np.zeros(S+1); p_const[0] = p0; p_const[-1] = pS

    # Precompute constant part: solve KKT with rhs = [0 ; Bp p_const + h]
    rhs2_const = (Bp @ p_const) + h
    rhs_const  = np.concatenate([np.zeros(H.shape[0]), np.asarray(rhs2_const).ravel()])
    sol_const  = KKT_lu.solve(rhs_const)
    a_const    = sol_const[:H.shape[0]]

    # Precompute columns of M: for each xi unit vector e_j
    M_cols = []
    B_free = Bp @ S_expand                      # (m x (S-1))
    for j in range(S-1):
        ej = np.zeros(S-1); ej[j] = 1.0
        rhs2 = B_free @ ej                      # only position rows contribute
        rhs  = np.concatenate([np.zeros(H.shape[0]), np.asarray(rhs2).ravel()])
        sol  = KKT_lu.solve(rhs)
        a_j  = sol[:H.shape[0]]
        M_cols.append(a_j)

    M = np.column_stack(M_cols) if M_cols else np.zeros((H.shape[0], 0))
    c = a_const

    # Return closures
    def coeffs_from_xi(xi: np.ndarray) -> np.ndarray:
        if xi.size == 0:
            return c.copy()
        return M @ xi + c

    return coeffs_from_xi, M, c, Q_blk


