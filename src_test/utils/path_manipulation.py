import numpy as np
from scipy.interpolate import interp1d

def simplify_path(path, keep_indices=None):
    if not path or len(path) < 2:
        return path

    def direction(p1, p2):
        return (p2[0] - p1[0], p2[1] - p1[1])

    knicks = [path[0]]
    prev_dir = direction(path[0], path[1])

    for i in range(1, len(path) - 1):
        curr_dir = direction(path[i], path[i + 1])
        if curr_dir != prev_dir:
            knicks.append(path[i])
        prev_dir = curr_dir

    knicks.append(path[-1])

    if keep_indices is None:
        return knicks

    max_index = len(knicks)
    resolved_indices = sorted(set(
        i if i >= 0 else max_index + i for i in keep_indices if -max_index <= i < max_index
    ))

    return [knicks[i] for i in resolved_indices]



def upsample_path(path_real, num_points):
    deltas = np.diff(path_real, axis=0)
    segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])

    target_lengths = np.linspace(0, cumulative_lengths[-1], num_points)

    interp_x = interp1d(cumulative_lengths, path_real[:, 0], kind='linear')
    interp_y = interp1d(cumulative_lengths, path_real[:, 1], kind='linear')

    upsampled_path = np.stack([interp_x(target_lengths), interp_y(target_lengths)], axis=1)
    return upsampled_path

def simplify_path_3d(path, keep_indices=None, mode="grid", tol=1e-9):
    if not path or len(path) < 2:
        return path

    P = np.asarray(path, dtype=float)
    n = len(P)

    def dir_grid(p1, p2):
        d = np.sign(p2 - p1)
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
            return True
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

    kinks = [path[0]]
    prev_dir = dfunc(P[0], P[1])

    for i in range(1, n - 1):
        curr_dir = dfunc(P[i], P[i + 1])
        if not sfunc(prev_dir, curr_dir):
            kinks.append(path[i])
        prev_dir = curr_dir

    kinks.append(path[-1])

    if keep_indices is None:
        return kinks

    m = len(kinks)
    resolved = sorted(set(
        i if i >= 0 else m + i
        for i in keep_indices
        if -m <= i < m
    ))
    return [kinks[i] for i in resolved]


def upsample_path_3d(path, num_points):
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
    fx = interp1d(cumlen, P[:, 0], kind='linear')
    fy = interp1d(cumlen, P[:, 1], kind='linear')
    fz = interp1d(cumlen, P[:, 2], kind='linear')
    Q = np.stack([fx(targets), fy(targets), fz(targets)], axis=1)
    return Q


def keep_turns_np(points_np: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_np)
    if pts.shape[0] <= 2:
        return pts.copy()

    steps = np.diff(pts, axis=0) 
    turns_mask = np.any(steps[1:] != steps[:-1], axis=1)
    keep_idx = np.concatenate([[0], 1 + np.flatnonzero(turns_mask), [pts.shape[0]-1]])
    return pts[keep_idx]

import numpy as np

def line_of_sight_free(grid: np.ndarray,
                       a_idx: tuple[int,int,int],
                       b_idx: tuple[int,int,int],
                       clearance: int = 0) -> bool:
    nx, ny, nz = grid.shape

    if clearance > 0:
        from numpy.lib.stride_tricks import sliding_window_view
        pad = clearance
        g = np.pad(grid, pad, mode="edge")
        win = sliding_window_view(g, (2*pad+1, 2*pad+1, 2*pad+1))
        grid = (win.any(axis=(3,4,5)))

    def inb(i,j,k): return (0<=i<nx and 0<=j<ny and 0<=k<nz)

    ax, ay, az = map(int, a_idx)
    bx, by, bz = map(int, b_idx)
    if not (inb(ax,ay,az) and inb(bx,by,bz)):
        return False
    if grid[ax,ay,az] or grid[bx,by,bz]:
        return False

    p0 = np.array([ax+0.5, ay+0.5, az+0.5], dtype=float)
    p1 = np.array([bx+0.5, by+0.5, bz+0.5], dtype=float)
    d  = p1 - p0

    if np.allclose(d, 0.0):
        return True

    vx, vy, vz = ax, ay, az

    step = np.sign(d).astype(int)
    step[abs(d) < 1e-15] = 0

    tMax = np.zeros(3, dtype=float)
    tDelta = np.empty(3, dtype=float)
    for i, (pi, di, vi, si) in enumerate(zip(p0, d, (vx,vy,vz), step)):
        if si > 0:
            next_boundary = vi + 1.0
            tMax[i] = (next_boundary - pi) / di
            tDelta[i] = 1.0 / di
        elif si < 0:
            next_boundary = vi * 1.0
            tMax[i] = (next_boundary - pi) / di
            tDelta[i] = -1.0 / di
        else:
            tMax[i] = np.inf
            tDelta[i] = np.inf

    while (vx,vy,vz) != (bx,by,bz):
        axis = int(np.argmin(tMax))
        if not np.isfinite(tMax[axis]):
            return False
        if axis == 0:
            vx += step[0]
        elif axis == 1:
            vy += step[1]
        else:
            vz += step[2]
        tMax[axis] += tDelta[axis]

        if not inb(vx,vy,vz):
            return False
        if grid[vx,vy,vz]:
            return False

    return True


def reduce_turns_by_los(turns_idx: np.ndarray,
                        grid: np.ndarray,
                        clearance: int = 0) -> np.ndarray:
    T = np.asarray(turns_idx, dtype=int)
    if T.ndim != 2 or T.shape[1] != 3 or T.shape[0] <= 2:
        return T

    kept = [T[0]]
    i = 0
    N = T.shape[0]

    while True:
        j = i + 1
        last_good = i + 1
        while j < N and line_of_sight_free(grid, tuple(T[i]), tuple(T[j]), clearance=clearance):
            last_good = j
            j += 1
        kept.append(T[last_good])
        if last_good == N - 1:
            break
        i = last_good

    return np.asarray(kept, dtype=int)


def sample_every_k(points, k):
    if not points:
        return []
    if k is None or k <= 1:
        return points[:]

    sampled = points[::k]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled