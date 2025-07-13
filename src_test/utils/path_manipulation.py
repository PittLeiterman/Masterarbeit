import numpy as np

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



