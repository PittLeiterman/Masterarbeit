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
