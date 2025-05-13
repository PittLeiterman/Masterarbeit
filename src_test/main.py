from input.trees2D import generate_forest, create_occupancy_grid, load_forest_from_file
from pathfinder.AStar import astar
from utils.path_manipulation import simplify_path
from optimization.preparation import (
    build_minimum_snap_qp,
    build_position_constraints,
    evaluate_polynomial_trajectory_time,
    build_continuity_constraints,
    build_initial_derivative_constraints,
)
from optimization.trajectory_admm_optimizer import run_admm_trajectory_optimization
from optimization.primal_step import admm_trajectory_opt, plot_segment_lengths

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
A_init, l_init, u_init = build_initial_derivative_constraints(
    path_real,
    velocity_start=np.array([0.0, 1.0]),
    velocity_end=np.array([0.0, 0.0])
)

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
traj_x, traj_y = evaluate_polynomial_trajectory_time(x_opt, num_points=60, poly_order=5)



# num_segments = len(path_real) - 1
# segment_times = np.ones(num_segments) * 1.0  # z. B. 1 Sekunde pro Segment

# x_primal = admm_trajectory_opt(
#     num_segments=num_segments,
#     segment_times=segment_times,
#     start=start_xy,
#     goal=goal_xy,
#     velocity_start=np.array([1.0, 0.0]),
#     velocity_end=np.array([1.0, 0.0]),
#     convex_regions=(A_list, b_list),  # <- Deine gültigen Regionen
#     max_iter=50,
#     tol=1e-3,
#     rho=1.0,
#     debug=True  # optional für Residuenanzeige
# )

# def evaluate_last_point(x_opt, segment_times, poly_order=7):
#     n_coeffs = poly_order + 1
#     seg = len(segment_times) - 1
#     T = segment_times[seg]

#     coeffs_x = x_opt[(seg * 2) * n_coeffs:(seg * 2 + 1) * n_coeffs].flatten()
#     coeffs_y = x_opt[(seg * 2 + 1) * n_coeffs:(seg * 2 + 2) * n_coeffs].flatten()

#     xT = np.polyval(coeffs_x[::-1], T)
#     yT = np.polyval(coeffs_y[::-1], T)

#     return xT, yT

# xT, yT = evaluate_last_point(x_primal, segment_times)
# print("Manuell berechneter Zielpunkt bei T:", (xT, yT))
# print("Zielpunkt (soll):", goal_xy)


# traj_x_primal, traj_y_primal = evaluate_polynomial_trajectory_time(x_primal, segment_times, poly_order=7)

# print("Zielpunkt (soll):", goal_xy)
# print("Trajektorien-Endpunkt (ist):", traj_x_primal[-1], traj_y_primal[-1])

initial_points = np.stack([traj_x, traj_y], axis=1)

# ADMM starten
optimized_points = run_admm_trajectory_optimization(
    initial_points,
    A_list=A_list,
    b_list=b_list,
    max_iter=20,
    rho=1.0,
    alpha=0.98
)



# ---------------------------------
# Plot
# ---------------------------------
ax = pdc.visualize_environment(Al=A_list, bl=b_list, p=path_real, planar=True)

# plot_segment_lengths(x_primal, segment_times, poly_order=7, ax=ax)

# Letzten Punkt aus Primal-Trajektorie anzeigen
# plt.plot(traj_x_primal[-1], traj_y_primal[-1], 'kx', markersize=10, label="Letzter Punkt (Primal)")


# Bäume einzeichnen
ax.plot(forest[:, 0], forest[:, 1], "o", color="green", label="Bäume")

# Originalpfad vorbereiten
path_array = np.array([(y, x) for (x, y) in path])

# Originalpfad einzeichnen
# ax.plot(path_array[:, 0], path_array[:, 1], '-', color="blue", linewidth=2, label="Originalpfad (A*)")

# Start und Ziel einzeichnen (optional, sieht gut aus)
# ax.plot(start_xy[0], start_xy[1], "rs", markersize=10, label="Start")
# ax.plot(goal_xy[0], goal_xy[1], "rs", markersize=10, label="Ziel")


# Originalpfad als Rasterpfad (volle Auflösung, viele Punkte)
# ax.plot(path_array[:, 0], path_array[:, 1], '--', color="orange", linewidth=1.5, label="A*-Pfad (voll)")

ax.plot(traj_x, traj_y, color="red", linestyle="-", linewidth=2, label="ADMM-Trajektorie")
# Zeitabgetastete Punkte markieren (z.B. als schwarze Punkte)
ax.plot(traj_x, traj_y, 'ko', markersize=3, label="Zeit-Stützstellen")

# Orangene gestrichelte Linie
ax.plot(optimized_points[:, 0], optimized_points[:, 1], linestyle='--', color='orange', linewidth=2, label="Optimierte Trajektorie (ADMM)")

# Kreuz-Marker an Punkten
ax.plot(optimized_points[:, 0], optimized_points[:, 1], 'x', color='orange', markersize=6)




# Legende und Plot anzeigen
ax.legend()
plt.show()
