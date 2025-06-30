import numpy as np

def simplify_path(path):
    if not path or len(path) < 3:
        return path

    def direction(p1, p2):
        return (p2[0] - p1[0], p2[1] - p1[1])

    knicks = [path[0]]  # Start immer behalten
    prev_dir = direction(path[0], path[1])

    for i in range(1, len(path) - 1):
        curr_dir = direction(path[i], path[i + 1])
        if curr_dir != prev_dir:
            knicks.append(path[i])
        prev_dir = curr_dir

    knicks.append(path[-1])  # Ziel immer behalten

    # Jetzt nur jede zweite Ecke behalten (außer Start und Ziel)
    reduced = [knicks[0]] + knicks[1:-1][::2] + [knicks[-1]]
    reduced.insert(2, knicks[2])  # Erste Ecke immer behalten
    for index in sorted([3, 2, 1], reverse=True):
        if len(reduced) > index:
            del reduced[index]
    return reduced


