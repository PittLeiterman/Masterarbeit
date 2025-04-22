import numpy as np

def simplify_path(path):
    if not path:
        return []
    
    simplified = [path[0]]  # erster Punkt immer
    # simplified.append(path[4])  # zweiter Punkt immer
    def direction(p1, p2):
        return (p2[0] - p1[0], p2[1] - p1[1])

    prev_dir = direction(path[0], path[1])

    for i in range(1, len(path)-1):
        curr_dir = direction(path[i], path[i+1])
        if curr_dir != prev_dir:
            simplified.append(path[i])
        prev_dir = curr_dir

    simplified.append(path[-1])  # letzter Punkt immer

    tep = [simplified[3]]
    tep.append(simplified[4])
    tep.append(simplified[5])
    tep.append(simplified[6])
    tep.append(simplified[7])
    tep.append(simplified[8])
    tep.append(simplified[9])
    tep.append(simplified[10])
    return tep
