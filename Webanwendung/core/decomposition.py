import numpy as np

def convex_decompose_and_clip(blocks, path, cols, rows):
    import pydecomp as pdc
    obstacles = np.array([[c+0.5,r+0.5] for (r,c) in sorted(blocks)], dtype=float)
    path_real = np.array([[c+0.5,r+0.5] for (r,c) in path], dtype=float)
    area_size = np.array([[float(cols), float(rows)]], dtype=float)
    return pdc.convex_decomposition_2D(obstacles, path_real, area_size)
