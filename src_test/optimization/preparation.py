import numpy as np
import math

def build_minimum_snap_qp(path_real):
    """
    Erzeugt die H- und g-Matrix für ein Minimum-Snap-QP durch gegebene Wegpunkte.
    
    path_real: ndarray (N+1, 2) — Wegpunkte als (x, y)
    Rückgabe: H (Quadratmatrix), g (Gradient)
    """
    n_segments = len(path_real) - 1
    n_coeff = 6  # quintic
    dim = 2  # x, y
    n_vars = n_segments * n_coeff * dim

    # Leere QP-Struktur
    H = np.zeros((n_vars, n_vars))
    g = np.zeros((n_vars, 1))

    # Snap-Kostenmatrix pro Segment (nur auf a4, a5)
    Q_snap = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 576, 1440],
        [0, 0, 0, 0, 1440, 4800]
    ])

    # Kosten über alle Segmente summieren
    for i in range(n_segments):
        for axis in range(dim):  # 0=x, 1=y
            idx = (i * dim + axis) * n_coeff
            H[idx:idx + n_coeff, idx:idx + n_coeff] = Q_snap

    return H, g


def print_qp_structure(H, g):
    print("H shape:", H.shape)
    print("g shape:", g.shape)
    print("Sparsity (non-zero entries):", np.count_nonzero(H))


def count_snap_cost(path_real):
    """ Hilfsfunktion: Anzahl der Snap-Kostenblöcke berechnen """
    n_segments = len(path_real) - 1
    return n_segments * 2  # 2 Achsen (x, y)

def build_position_constraints(path_real):
    """
    Erzeugt Gleichungs-Constraints für Positionspassung an Segmentgrenzen.
    Gibt A_eq, l_eq, u_eq zurück (für QP-Solver).
    """
    n_segments = len(path_real) - 1
    n_coeff = 6
    dim = 2  # x, y
    n_vars = n_segments * n_coeff * dim

    # Für jede Segmentgrenze zwei Gleichungen: x(t=0)=..., x(t=1)=... und y analog
    n_eqs = 2 * dim * n_segments
    A_eq = np.zeros((n_eqs, n_vars))
    b_eq = np.zeros((n_eqs, 1))

    eval_t0 = np.array([1, 0, 0, 0, 0, 0])
    eval_t1 = np.array([1, 1, 1, 1, 1, 1])

    row = 0
    for i in range(n_segments):
        p0 = path_real[i]
        p1 = path_real[i + 1]

        for axis in range(dim):  # 0=x, 1=y
            idx = (i * dim + axis) * n_coeff

            # Startpunkt
            A_eq[row, idx:idx + n_coeff] = eval_t0
            b_eq[row, 0] = p0[axis]
            row += 1

            # Endpunkt
            A_eq[row, idx:idx + n_coeff] = eval_t1
            b_eq[row, 0] = p1[axis]
            row += 1

    return A_eq, b_eq, b_eq  # l == u == b

def evaluate_polynomial_trajectory_time(x_opt, num_points=300, poly_order=5):
    """
    Evaluates the entire multi-segment trajectory over a fixed global time axis [0, N_segments].
    Returns (x_vals, y_vals) each with shape (num_points,)
    """
    n_coeff = poly_order + 1
    n_segments = len(x_opt) // (2 * n_coeff)
    total_time = n_segments  # 1 sec per segment → global t ∈ [0, n_segments]

    ts = np.linspace(0, total_time, num_points)
    traj_x, traj_y = [], []

    for t in ts:
        seg_idx = min(int(t), n_segments - 1)  # Sicherstellen, dass Index nicht überläuft
        local_t = t - seg_idx  # Lokale Zeit in Segment ∈ [0, 1]

        idx_x = (seg_idx * 2) * n_coeff
        idx_y = (seg_idx * 2 + 1) * n_coeff

        coeffs_x = x_opt[idx_x:idx_x + n_coeff].flatten()
        coeffs_y = x_opt[idx_y:idx_y + n_coeff].flatten()

        powers = np.array([local_t**j for j in range(n_coeff)])
        traj_x.append(np.dot(coeffs_x, powers))
        traj_y.append(np.dot(coeffs_y, powers))

    return np.array(traj_x), np.array(traj_y)







def build_continuity_constraints(path_real, order=3):
    """
    Baut Gleichungs-Constraints für C1–C3-Kontinuität zwischen Segmenten.
    order = 1 (Geschw.), 2 (Beschleunigung), 3 (Snap)
    """
    assert 1 <= order <= 3

    n_segments = len(path_real) - 1
    n_coeff = 6
    dim = 2  # x, y
    n_vars = n_segments * n_coeff * dim

    constraints = []

    # Ableitungsbasis bei t=0 und t=1
    def deriv_basis(t, d):
        return np.array([
            0 if i < d else math.factorial(i) / math.factorial(i - d) * t**(i - d)
            for i in range(n_coeff)
        ])

    for i in range(n_segments - 1):  # Übergänge zwischen Segment i → i+1
        for axis in range(dim):
            base1 = (i * dim + axis) * n_coeff
            base2 = ((i + 1) * dim + axis) * n_coeff

            for d in range(1, order + 1):  # d = 1 (v), 2 (a), 3 (snap)
                row = np.zeros(n_vars)
                row[base1:base1 + n_coeff] = deriv_basis(1.0, d)
                row[base2:base2 + n_coeff] = -deriv_basis(0.0, d)
                constraints.append(row)

    A_cont = np.vstack(constraints)
    b_cont = np.zeros((A_cont.shape[0], 1))
    return A_cont, b_cont, b_cont

def build_initial_derivative_constraints(path_real, velocity_start=None, velocity_end=None):
    """
    Setzt Start- und/oder Endgeschwindigkeit als Gleichungen.
    """
    n_coeff = 6
    dim = 2
    n_segments = len(path_real) - 1
    n_vars = n_segments * n_coeff * dim

    constraints = []
    rhs = []

    if velocity_start is not None:
        deriv1_basis_start = np.array([0, 1, 0, 0, 0, 0])
        for axis in range(dim):
            row = np.zeros(n_vars)
            idx = axis * n_coeff
            row[idx:idx + n_coeff] = deriv1_basis_start
            constraints.append(row)
            rhs.append(velocity_start[axis])

    if velocity_end is not None:
        deriv1_basis_end = np.array([0, 1, 2, 3, 4, 5])  # t=1 abgeleitet
        for axis in range(dim):
            row = np.zeros(n_vars)
            idx = ((n_segments - 1) * dim + axis) * n_coeff
            row[idx:idx + n_coeff] = deriv1_basis_end
            constraints.append(row)
            rhs.append(velocity_end[axis])

    A = np.vstack(constraints)
    b = np.array(rhs).reshape(-1, 1)
    return A, b, b

