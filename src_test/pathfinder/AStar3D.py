# astar_3d.py
from __future__ import annotations
import heapq
from typing import Dict, List, Optional, Tuple
import numpy as np
import pyvista as pv

from input.make3DObstacles import load_voxel_grid

# ==== HARD-CODED SETTINGS=======================================
VOXEL_FILE = "input/data/shape_3d_1.txt"
PADDING = 0                
CONNECTIVITY = 6           # 6, 18, or 26
TUBE_RADIUS = 0.2          # visual thickness of path
# ============================================================================


def split_shell_inner(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (shell_mask, inner_mask) for a boolean occupancy grid.
    shell: occupied voxels that touch any free voxel in 6-neighborhood
    inner: occupied voxels fully surrounded by occupied voxels in 6-neighborhood
    """
    g = grid
    p = np.pad(g, 1, constant_values=False)
    n_all = (
        p[:-2, 1:-1, 1:-1] &
        p[2:,  1:-1, 1:-1] &
        p[1:-1, :-2,  1:-1] &
        p[1:-1, 2:,   1:-1] &
        p[1:-1, 1:-1, :-2 ] &
        p[1:-1, 1:-1, 2:  ]
    )
    inner = g & n_all
    shell = g & (~n_all)
    return shell, inner


# ---------- neighbors for A* ----------
def neighbor_steps(connectivity: int) -> List[Tuple[int, int, int]]:
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26")
    steps = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    if connectivity >= 18:
        steps += [
            (1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
            (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),
            (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1),
        ]
    if connectivity == 26:
        steps += [
            (a,b,c)
            for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1)
            if not (a==0 and b==0 and c==0) and (a,b,c) not in steps
        ]
    return steps


def heuristic(a: Tuple[int,int,int], b: Tuple[int,int,int], connectivity: int) -> float:
    if connectivity == 6:
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])
    dx, dy, dz = (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    return (dx*dx + dy*dy + dz*dz) ** 0.5


def astar_3d(grid: np.ndarray,
             start_idx: Tuple[int,int,int],
             goal_idx: Tuple[int,int,int],
             connectivity: int = 6) -> Optional[List[Tuple[int,int,int]]]:
    nx, ny, nz = grid.shape
    in_bounds = lambda n: (0 <= n[0] < nx and 0 <= n[1] < ny and 0 <= n[2] < nz)
    is_free = lambda n: in_bounds(n) and (not grid[n])

    if not is_free(start_idx) or not is_free(goal_idx):
        return None

    steps = neighbor_steps(connectivity)

    def step_cost(d):
        ax = abs(d[0]) + abs(d[1]) + abs(d[2])
        return 1.0 if ax == 1 else (2**0.5 if ax == 2 else 3**0.5)

    def traversable(cur, d):
        n = (cur[0]+d[0], cur[1]+d[1], cur[2]+d[2])
        if not is_free(n):
            return False
        ax = abs(d[0]) + abs(d[1]) + abs(d[2])
        if ax >= 2:
            checks = []
            if d[0] != 0: checks.append((cur[0]+d[0], cur[1], cur[2]))
            if d[1] != 0: checks.append((cur[0], cur[1]+d[1], cur[2]))
            if d[2] != 0: checks.append((cur[0], cur[1], cur[2]+d[2]))
            for c in checks:
                if not is_free(c):
                    return False
        return True

    open_heap = []
    heapq.heappush(open_heap, (0.0, start_idx))
    g_score = {start_idx: 0.0}
    came_from = {}
    f_score = {start_idx: heuristic(start_idx, goal_idx, connectivity)}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal_idx:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        gi = g_score[current]
        for d in steps:
            if not traversable(current, d):
                continue
            n = (current[0]+d[0], current[1]+d[1], current[2]+d[2])
            cand = gi + step_cost(d)
            if cand < g_score.get(n, float("inf")):
                came_from[n] = current
                g_score[n] = cand
                f = cand + heuristic(n, goal_idx, connectivity)
                if f < f_score.get(n, float("inf")):
                    f_score[n] = f
                    heapq.heappush(open_heap, (f, n))
    return None



# ---------- Visualization with PyVista ----------
def make_surface_from_mask(mask: np.ndarray,
                           origin: Tuple[int,int,int],
                           spacing=(1,1,1)) -> pv.PolyData:
    nx, ny, nz = mask.shape
    ox, oy, oz = origin
    img = pv.ImageData(dimensions=(nx+1, ny+1, nz+1), spacing=spacing, origin=(ox, oy, oz))
    img.cell_data["occ"] = mask.astype(np.uint8).ravel(order="F")
    vol = img.threshold(0.5, scalars="occ")
    return vol.extract_surface()


def visualize(vg, path_coords: Optional[List[Tuple[int,int,int]]]):
    shell, inner = split_shell_inner(vg.grid)

    surf_shell = make_surface_from_mask(shell, vg.info.origin)
    surf_inner = make_surface_from_mask(inner, vg.info.origin) if inner.any() else None

    p = pv.Plotter()
    # Grid shells
    p.add_mesh(surf_shell, color="blue", opacity=0.25, show_edges=False)
    if surf_inner is not None:
        p.add_mesh(surf_inner, color=(0.0, 0.0, 0.6), opacity=0.95, show_edges=False)

    # Path overlay
    if path_coords and len(path_coords) >= 2:
        centers = np.array(path_coords, dtype=float) + 0.5
        centers_world = np.array([vg.info.to_coord(tuple(map(int, c))) for c in centers])
        centers_world = centers_world + 0.5

        line = pv.Spline(centers_world, n_points=len(centers_world))
        tube = line.tube(radius=TUBE_RADIUS, n_sides=16)
        p.add_mesh(tube, color="red", opacity=1.0)

        start_c = np.array(vg.info.to_coord(path_coords[0])) + 0.5
        goal_c  = np.array(vg.info.to_coord(path_coords[-1])) + 0.5
        p.add_mesh(pv.Sphere(radius=TUBE_RADIUS*1.6, center=start_c), color="green")
        p.add_mesh(pv.Sphere(radius=TUBE_RADIUS*1.6, center=goal_c),  color="orange")
    else:
        print("No path found (or too short). Showing grid only.")

    p.show_axes()
    p.show()


def main():
    return


if __name__ == "__main__":
    main()
