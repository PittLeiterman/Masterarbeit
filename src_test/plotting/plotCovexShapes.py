import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from input.trees2D import generate_forest, create_occupancy_grid, load_forest_from_file
from pathfinder.AStar import astar
from utils.path_manipulation import simplify_path
import pydecomp as pdc
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from scipy.interpolate import splprep, splev

# Parameter
n_trees = 3
area_size = (25, 25)
min_distance = 2.0

# Baum- und Grid-Erzeugung
# forest = generate_forest(n_trees, area_size, min_distance)
base_dir = Path(__file__).resolve().parent.parent  # geht zwei Ebenen hoch: aus /plotting raus
forest_path = base_dir / "input" / "data" / "my_trees.txt"

forest = load_forest_from_file(str(forest_path))
grid = create_occupancy_grid(forest, area_size, area_size, tree_size=1)

print(grid)
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

plt.imshow(grid, cmap="gray_r", origin="lower", extent=[0, area_size[0], 0, area_size[1]], alpha=0.3)


# Konvexe Zerlegung
A_list, b_list = pdc.convex_decomposition_2D(forest, path_real, np.array([area_size]))

# ---------------------------------
# Plot
# ---------------------------------
ax = pdc.visualize_environment(Al=A_list, bl=b_list, p=path_real, planar=True)

# Bäume einzeichnen
ax.plot(forest[:, 0], forest[:, 1], "o", color="green", label="Bäume")

# Originalpfad vorbereiten
path_array = np.array([(y, x) for (x, y) in path])

# Originalpfad einzeichnen
ax.plot(path_array[:, 0], path_array[:, 1], '-', color="blue", linewidth=2, label="Originalpfad (A*)")

# Start und Ziel einzeichnen (optional, sieht gut aus)
ax.plot(start_xy[0], start_xy[1], "rs", markersize=10, label="Start")
ax.plot(goal_xy[0], goal_xy[1], "rs", markersize=10, label="Ziel")


# Originalpfad als Rasterpfad (volle Auflösung, viele Punkte)
ax.plot(path_array[:, 0], path_array[:, 1], '--', color="orange", linewidth=1.5, label="A*-Pfad (voll)")


# 1. Parametric spline (t) from simplified path
x, y = path_real[:, 0], path_real[:, 1]
tck, u = splprep([x, y], s=0.5, k=3)  # s controls smoothness, k=3 is cubic

# 2. Evaluate spline at fine-grained points
u_fine = np.linspace(0, 1, 300)
x_smooth, y_smooth = splev(u_fine, tck)

# 3. Plot the smooth curve
ax.plot(x_smooth, y_smooth, 'r--', linewidth=2, label="Glatter Pfad (Spline)")

# Legende und Plot anzeigen
ax.legend()
plt.show()
