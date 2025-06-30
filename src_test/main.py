from input.trees2D import create_occupancy_grid, load_forest_from_file
from pathfinder.AStar import astar
from utils.path_manipulation import simplify_path

from optimization.primal_step import minimum_snap_trajectory, evaluate_polynomial
from optimization.projection_utils import project_segments_to_convex_regions

import pydecomp as pdc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Parameter
n_trees = 3
area_size = (25, 25)
min_distance = 2.0

# Baum- und Grid-Erzeugung
# forest = generate_forest(n_trees, area_size, min_distance)
forest = load_forest_from_file("input/data/my_trees.txt")
grid = create_occupancy_grid(forest, area_size, area_size, tree_size=1)

# Start- und Zielpunkt
# Original (row, col)
start = (3, 3)
goal = (area_size[0] - 3, 3)

# Umgedreht (x, y) → für pydecomp & plotting
start_xy = (start[1], start[0])
goal_xy = (goal[1], goal[0])

# A* Pfadsuche
path = astar(grid, start, goal)

# Pfad prüfen
if path is None or len(path) == 0:
    print("Kein Pfad gefunden!")
    exit()

# Pfad Vereinfachung
path_simplified = simplify_path(path)

# Für pydecomp
path_real = np.array([(y, x) for (x, y) in path_simplified])


# Konvexe Zerlegung
A_list, b_list = pdc.convex_decomposition_2D(forest, path_real, np.array([area_size]))

ax = pdc.visualize_environment(Al=A_list, bl=b_list, p=path_real, planar=True)

ax.plot(start_xy[0], start_xy[1], 'go', label='Start')
ax.plot(goal_xy[0], goal_xy[1], 'bo', label='Goal')

ax.plot(forest[:, 0], forest[:, 1], "o", color="green", label="Bäume")


# --- Initialisierung ---
v_start = (0.0, 0.0)  # Starting at rest
v_end = (0.0, 0.0)    # Ending at rest

# Generate trajectory coefficients
coeffs_x, coeffs_y, segment_times = minimum_snap_trajectory(start_xy, goal_xy, v_start, v_end, num_segments=15)

projected_x, projected_y = project_segments_to_convex_regions(
    coeffs_x, coeffs_y, segment_times, A_list, b_list
)

colors = cm.viridis(np.linspace(0, 1, len(segment_times) - 1))

trajectory_x = []
trajectory_y = []
segment_lengths = []

for i in range(len(segment_times) - 1):
    t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], 50)
    a_x = coeffs_x[i].value
    a_y = coeffs_y[i].value

    x_vals = [evaluate_polynomial(a_x, t) for t in t_vals]
    y_vals = [evaluate_polynomial(a_y, t) for t in t_vals]

    # Geschwindigkeit berechnen
    dx_vals = [sum(c * t**i for i, c in enumerate(np.polyder(a_x[::-1]))) for t in t_vals]
    dy_vals = [sum(c * t**i for i, c in enumerate(np.polyder(a_y[::-1]))) for t in t_vals]
    speeds = [np.hypot(dx, dy) for dx, dy in zip(dx_vals, dy_vals)]
    
    ax.plot(x_vals, y_vals, color=colors[i], label=f'Segment {i+1}')
    dist = sum(np.hypot(np.diff(x_vals), np.diff(y_vals)))
    segment_lengths.append(dist)

for x_vals, y_vals in zip(projected_x, projected_y):
    ax.plot(x_vals, y_vals, 'k--', linewidth=1.5, label="projiziert" if 'projiziert' not in ax.get_legend_handles_labels()[1] else "")
# a_x = coeffs_x[4].value
# a_y = coeffs_y[4].value

# x_vals = [evaluate_polynomial(a_x, t) for t in t_vals]
# y_vals = [evaluate_polynomial(a_y, t) for t in t_vals]
# x_vals_shifted = [x + 2.0 for x in x_vals]
# ax.plot(x_vals_shifted, y_vals, color='red', linewidth=3, label='Segment 8 (verschoben)')
# ax.plot(x_vals_shifted[0], y_vals[0], 'ro')   # Startpunkt
# ax.plot(x_vals_shifted[-1], y_vals[-1], 'rx')


ax.set_aspect('equal')
ax.grid(True)
plt.show()


plt.figure()
plt.bar(range(len(segment_lengths)), segment_lengths)
plt.title("Räumliche Länge pro Segment")
plt.xlabel("Segmentindex")
plt.ylabel("Strecke")
plt.show()
