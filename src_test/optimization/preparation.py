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

def evaluate_polynomial_trajectory(x_opt, path_real, num_points_per_segment=50):
    """
    Gibt x,y-Werte der geplanten Trajektorie zurück, die aus den optimierten Koeffizienten evaluiert wurde.
    
    x_opt: Ergebnis aus QP-Solver (Koeffizientenvektor)
    path_real: Original-Wegpunkte (zur Segmentanzahl)
    Rückgabe: X, Y als 1D-Arrays
    """
    n_segments = len(path_real) - 1
    n_coeff = 6
    dim = 2  # x, y

    traj_x = []
    traj_y = []

    for i in range(n_segments):
        # Indizes im großen Koeff-Vektor
        idx_x = (i * dim + 0) * n_coeff
        idx_y = (i * dim + 1) * n_coeff
        coeffs_x = x_opt[idx_x:idx_x + n_coeff].flatten()
        coeffs_y = x_opt[idx_y:idx_y + n_coeff].flatten()

        t_vals = np.linspace(0, 1, num_points_per_segment)
        for t in t_vals:
            powers = np.array([t**i for i in range(n_coeff)])
            x = np.dot(coeffs_x, powers)
            y = np.dot(coeffs_y, powers)
            traj_x.append(x)
            traj_y.append(y)

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

def build_initial_derivative_constraints(path_real, velocity=None):
    """
    Fixiert die Richtung (1. Ableitung) am Startpunkt, optional auch am Ende.
    velocity: np.array([vx, vy]) oder None
    """
    n_coeff = 6
    dim = 2
    n_segments = len(path_real) - 1
    n_vars = n_segments * n_coeff * dim

    if velocity is None:
        velocity = np.array([0.0, 0.0])

    A = np.zeros((2, n_vars))
    b = np.zeros((2, 1))

    deriv1_basis = np.array([0, 1, 0, 0, 0, 0])  # Ableitung bei t = 0

    for axis in range(dim):  # 0=x, 1=y
        idx = axis * n_coeff
        A[axis, idx:idx + n_coeff] = deriv1_basis
        b[axis, 0] = velocity[axis]

    return A, b, b
