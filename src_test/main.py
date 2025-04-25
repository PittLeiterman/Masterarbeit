from input.trees2D import generate_forest, create_occupancy_grid, load_forest_from_file
from pathfinder.AStar import astar
from utils.path_manipulation import simplify_path
from optimization.preparation import (
    build_minimum_snap_qp,
    build_position_constraints,
    evaluate_polynomial_trajectory,
    build_continuity_constraints,
    build_initial_derivative_constraints,
)

from optimization.admm import (
    QPProb,
    admm_solve,
)
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

# -------------------
# 1. Minimum-Snap QP erzeugen
# -------------------
H, g = build_minimum_snap_qp(path_real)

# -------------------
# 2. Gleichungen aufbauen
#    - Positionszwang an Start/Ende jedes Segments
#    - Glätte (C1–C3) an Übergängen
# -------------------
A_pos, l_pos, u_pos = build_position_constraints(path_real)
A_cont, l_cont, u_cont = build_continuity_constraints(path_real, order=3)
A_init, l_init, u_init = build_initial_derivative_constraints(path_real, velocity=np.array([1.0, 0.0]))

# -------------------
# 3. Alle Constraints zusammenführen
# -------------------
A = np.vstack([A_pos, A_cont, A_init])
l = np.vstack([l_pos, l_cont, l_init])
u = np.vstack([u_pos, u_cont, u_init])

# -------------------
# 4. ADMM-Problem instanziieren
# -------------------
prob = QPProb(H, g, A, l, u)

# -------------------
# 5. Lösung berechnen
# -------------------
x_opt, _, _ = admm_solve(prob, tol=1e-3, max_iter=1000)

# Werte berechnen
traj_x, traj_y = evaluate_polynomial_trajectory(x_opt, path_real)

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

ax.plot(traj_x, traj_y, color="red", linestyle="-", linewidth=2, label="ADMM-Trajektorie")


# Legende und Plot anzeigen
ax.legend()
plt.show()
