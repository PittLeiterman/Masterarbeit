from input.trees2D import create_occupancy_grid, load_forest_from_file
from pathfinder.AStar import astar
from utils.path_manipulation import simplify_path
import pydecomp as pdc

import numpy as np
import matplotlib.pyplot as plt

area_size = (50, 50)
min_distance = 1.0

# Baum- und Grid-Erzeugung
forest = load_forest_from_file("input/data/u.txt", False)
grid = create_occupancy_grid(forest, area_size, area_size, tree_size=1)

# Start- und Zielpunkt
# Original (row, col)
start = (15, 35)
goal = (5, 44)


# Umgedreht (x, y) → für pydecomp & plotting
start_xy = (start[1], start[0])
goal_xy = (goal[1], goal[0])


# A* Pfadsuche
#path = astar(grid, start, goal)
path = [(15, 35), (33, 17), (44, 25), (37, 45), (5, 44)]

print(f"Gefundener Pfad: {path}")

# Für pydecomp
path_real = np.array([(y, x) for (x, y) in path])


# Konvexe Zerlegung
A_list, b_list = pdc.convex_decomposition_2D(forest, path_real, np.array([area_size]))

ax = pdc.visualize_environment(Al=A_list, bl=b_list, p=path_real, planar=True)
ax.plot(forest[:, 0], forest[:, 1], "o", color="green", label="Bäume")
ax.legend()
plt.show()