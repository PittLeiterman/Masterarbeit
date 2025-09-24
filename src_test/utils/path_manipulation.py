import numpy as np
from scipy.interpolate import interp1d

def simplify_path(path, keep_indices=None):
    """
    Vereinfacht den Pfad, indem nur definierte Ecken (Knickpunkte) behalten werden.
    
    Args:
        path (list of (row, col)): Originalpfad aus A*
        keep_indices (list of int): Liste der Indizes der zu behaltenden Ecken (bezogen auf erkannte Ecken)

    Returns:
        list of (row, col): Vereinfachter Pfad
    """
    if not path or len(path) < 2:
        return path

    def direction(p1, p2):
        return (p2[0] - p1[0], p2[1] - p1[1])

    # Alle Knickpunkte sammeln (Start und Ziel inklusive)
    knicks = [path[0]]
    prev_dir = direction(path[0], path[1])

    for i in range(1, len(path) - 1):
        curr_dir = direction(path[i], path[i + 1])
        if curr_dir != prev_dir:
            knicks.append(path[i])
        prev_dir = curr_dir

    knicks.append(path[-1])

    if keep_indices is None:
        return knicks  # Keine Auswahl, gib alle Ecken zurück

    # Indizes bereinigen (negativ zu positiv, innerhalb der Liste)
    max_index = len(knicks)
    resolved_indices = sorted(set(
        i if i >= 0 else max_index + i for i in keep_indices if -max_index <= i < max_index
    ))

    return [knicks[i] for i in resolved_indices]



def upsample_path(path_real, num_points):
    # Schritt 1: Berechne kumulative Distanzen (Weglängen)
    deltas = np.diff(path_real, axis=0)
    segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])

    # Schritt 2: Erzeuge neue gleichmäßig verteilte Längenwerte
    target_lengths = np.linspace(0, cumulative_lengths[-1], num_points)

    # Schritt 3: Interpolation entlang der Pfadlänge
    interp_x = interp1d(cumulative_lengths, path_real[:, 0], kind='linear')
    interp_y = interp1d(cumulative_lengths, path_real[:, 1], kind='linear')

    upsampled_path = np.stack([interp_x(target_lengths), interp_y(target_lengths)], axis=1)
    return upsampled_path

def simplify_path_3d(path, keep_indices=None, mode="grid", tol=1e-9):
    """
    Simplify a 3D path by keeping only direction changes (kinks).
    - path: list/array of (x,y,z) points (ints for grid indices or floats for world coords)
    - keep_indices: list of indices into the *kink list* (start and end are included).
        Negative indices allowed (Python-style).
        If None -> return all kinks.
    - mode:
        "grid": two consecutive segments are 'same direction' iff sign(diff) is identical
                (works for 6/18/26-connectivity integer paths).
        "geom":  two segments are 'collinear & same orientation' if cross product ~ 0 and
                 dot product > 0 (tolerance = tol).
    Returns: list of 3D points (same type as input) corresponding to the kept kinks.
    """
    if not path or len(path) < 2:
        return path

    P = np.asarray(path, dtype=float)  # safe for both int/float; we’ll output original points
    n = len(P)

    def dir_grid(p1, p2):
        d = np.sign(p2 - p1)   # each component in {-1,0,1}
        return tuple(d.astype(int))

    def same_dir_grid(d1, d2):
        return d1 == d2

    def dir_geom(p1, p2):
        d = p2 - p1
        nrm = np.linalg.norm(d)
        if nrm == 0:
            return None
        return d / nrm

    def same_dir_geom(u, v, tol=1e-9):
        if u is None or v is None:
            return True  # degenerate zero segment: ignore as no change
        # Collinearity: |u x v| ~ 0 and same orientation: u·v > 0
        cross = np.linalg.norm(np.cross(u, v))
        dot   = float(np.dot(u, v))
        return cross <= tol and dot > 0

    # choose direction functions
    if mode == "grid":
        dfunc = dir_grid
        sfunc = same_dir_grid
    elif mode == "geom":
        dfunc = dir_geom
        sfunc = lambda a,b: same_dir_geom(a,b,tol=tol)
    else:
        raise ValueError('mode must be "grid" or "geom"')

    # collect kinks (always include first and last)
    kinks = [path[0]]
    prev_dir = dfunc(P[0], P[1])

    for i in range(1, n - 1):
        curr_dir = dfunc(P[i], P[i + 1])
        if not sfunc(prev_dir, curr_dir):
            kinks.append(path[i])  # keep original datatype/tuple if given
        prev_dir = curr_dir

    kinks.append(path[-1])

    if keep_indices is None:
        return kinks

    # normalize keep indices (support negatives, clamp to range)
    m = len(kinks)
    resolved = sorted(set(
        i if i >= 0 else m + i
        for i in keep_indices
        if -m <= i < m
    ))
    return [kinks[i] for i in resolved]


def upsample_path_3d(path, num_points):
    """
    Evenly resample a 3D polyline to 'num_points' points by arc-length.
    - path: array-like of shape (N, 3)
    - num_points: int >= 2
    Returns: ndarray (num_points, 3)
    """
    P = np.asarray(path, dtype=float)
    if len(P) == 0:
        return P
    if len(P) == 1:
        return np.repeat(P, num_points, axis=0)

    deltas = np.diff(P, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumlen[-1]
    if total == 0:
        return np.repeat(P[:1], num_points, axis=0)

    targets = np.linspace(0.0, total, num_points)
    # interpolate each coordinate independently along arc-length
    fx = interp1d(cumlen, P[:, 0], kind='linear')
    fy = interp1d(cumlen, P[:, 1], kind='linear')
    fz = interp1d(cumlen, P[:, 2], kind='linear')
    Q = np.stack([fx(targets), fy(targets), fz(targets)], axis=1)
    return Q


import numpy as np
from typing import List, Sequence

def _unique_consecutive(points: Sequence) -> (List, List[int]):
    """
    Entfernt direkt aufeinanderfolgende Duplikate, liefert (gefilterte_punkte, index_mapping).
    index_mapping[i] = Originalindex im Eingabepfad für den i-ten gefilterten Punkt.
    """
    out, idx_map = [], []
    prev = None
    for k, p in enumerate(points):
        tup = tuple(p)
        if tup != prev:
            out.append(p)
            idx_map.append(k)
            prev = tup
    return out, idx_map

import numpy as np

def keep_turns_np(points_np: np.ndarray) -> np.ndarray:
    """
    points_np: shape (N,3), int (Gridindices)
    Rückgabe: shape (M,3) mit Knicken inkl. erster/letzter Punkt
    """
    pts = np.asarray(points_np)
    if pts.shape[0] <= 2:
        return pts.copy()

    steps = np.diff(pts, axis=0)  # (N-1, 3), Werte in {-1,0,1}
    # Ein Knick an Index i (Punkt i) liegt vor, wenn steps[i] != steps[i-1]
    turns_mask = np.any(steps[1:] != steps[:-1], axis=1)  # (N-2,)
    # Wir nehmen immer ersten & letzten Punkt, plus die i, wo turns_mask True ist
    keep_idx = np.concatenate([[0], 1 + np.flatnonzero(turns_mask), [pts.shape[0]-1]])
    return pts[keep_idx]

