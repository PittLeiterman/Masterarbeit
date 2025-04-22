# pathfinder/AStar.py

import numpy as np
import heapq

def heuristic(a, b):
    # Euklidische Distanz
    return np.linalg.norm(np.array(a) - np.array(b))

def get_neighbors(node, grid):
    rows, cols = grid.shape
    r, c = node
    neighbors = []
    
    # 8 Richtungen (inkl. Diagonalen)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for dr, dc in moves:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            if abs(dr) + abs(dc) == 2:
                # Diagonal: Prüfen ob angrenzende Kacheln frei sind (no corner cutting)
                if grid[r, cc] == 1 or grid[rr, c] == 1:
                    continue
            neighbors.append((rr, cc))
    return neighbors

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)

        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        for neighbor in get_neighbors(current, grid):
            r, c = neighbor
            if grid[r, c] == 1:  # Hindernis
                continue
            if neighbor in visited:
                continue

            heapq.heappush(open_set, (g + heuristic(neighbor, goal), g + 1, neighbor, path + [neighbor]))
    return None
