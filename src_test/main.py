from input.trees2D import create_occupancy_grid, load_forest_from_file
from pathfinder.AStar import astar
from utils.path_manipulation import simplify_path
from optimization.bezier_primal_step import generate_fixed_bezier_segments, optimize_primal_step, bezier_matrix, M, snap_cost_matrix

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
traj_x_plt, traj_y_plt = evaluate_polynomial_trajectory_time(x_opt, num_points=100, poly_order=5)



# Visualisierung der Umgebung
ax = pdc.visualize_environment(Al=A_list, bl=b_list, p=path_real, planar=True)

# Bäume einzeichnen
ax.plot(forest[:, 0], forest[:, 1], "o", color="green", label="Bäume")

#ax.plot(traj_x_plt, traj_y_plt, color="red", linestyle="-", linewidth=2, label="Original Trajektorie")


n_segments = 10  # kannst du frei wählen
original_segments = generate_fixed_bezier_segments(x_opt, n_segments=n_segments, poly_order=5, bezier_order=M)
optimized_segments = optimize_primal_step(original_segments, snap_cost_matrix(M))

# === Bézier-Basis vorbereiten ===
t_vals = np.linspace(0, 1, 100)
B = bezier_matrix(M, t_vals)

colors = ['green', 'yellow']
labels = ['$X_i$ (primal)', '$\\bar{X}_i$ (slack)']

for i, (X_orig, X_slack) in enumerate(zip(original_segments, optimized_segments)):
    X_orig = X_orig.squeeze()
    X_slack = X_slack.squeeze()

    curve_orig = B @ X_orig
    curve_opt = B @ X_slack

    # Nur einmal legend label setzen
    label_orig = labels[0] if i == 0 else None
    label_slack = labels[1] if i == 0 else None

    # Ursprüngliche Bézier-Kurve
    ax.plot(curve_orig[:, 0], curve_orig[:, 1], color=colors[0], linestyle="--", alpha=0.7, label=label_orig)

    # Optimierte Slack-Kurve
    ax.plot(curve_opt[:, 0], curve_opt[:, 1], color=colors[1], linestyle="-", alpha=0.8, label=label_slack)


ax.set_title("Vergleich von $X_i$ (primal) und $\\bar{X}_i$ (slack)")
ax.set_aspect('equal')
ax.legend()
plt.show()

