from input.trees2D import create_occupancy_grid, load_forest_from_file
from pathfinder.AStar import astar
from utils.path_manipulation import simplify_path

from optimization.primal_step import minimum_snap_trajectory, evaluate_polynomial, solve_primal_step
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
coeffs_x, coeffs_y, segment_times = minimum_snap_trajectory(start_xy, goal_xy, v_start, v_end, num_segments=30)

projected_x, projected_y = project_segments_to_convex_regions(
    coeffs_x, coeffs_y, segment_times, A_list, b_list
)

colors = cm.viridis(np.linspace(0, 1, len(segment_times) - 1))

trajectory_x = []
trajectory_y = []
segment_lengths = []

for i in range(len(segment_times) - 1):
    t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], 30)
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
    ax.plot(x_vals, y_vals, 'r-', linewidth=2, label="projiziert" if 'projiziert' not in ax.get_legend_handles_labels()[1] else "")

plt.show()

rho = 2000.0
max_iters = 200
eps = 0.3
beta = 0.1

# Dummy-Initialisierung für z und u
z_traj = [np.column_stack((x, y)) for x, y in zip(projected_x, projected_y)]

# Erste Iteration: berechne u^1
x_traj = []  # evaluate aktuelle x-Trajektorie von coeffs_x/y
for i in range(len(segment_times) - 1):
    t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], 30)
    a_x = coeffs_x[i].value
    a_y = coeffs_y[i].value
    x_vals = [evaluate_polynomial(a_x, t) for t in t_vals]
    y_vals = [evaluate_polynomial(a_y, t) for t in t_vals]
    x_traj.append(np.column_stack((x_vals, y_vals)))

u_traj = [x - z for x, z in zip(x_traj, z_traj)]  # u^1

for k in range(max_iters):
    print(f"--- Iteration {k+1} ---")
    if k == 0:
        z_traj_prev = z_traj.copy()
    if k > 0:
        z_traj_extrapolated = [
            z_curr + beta * (z_curr - z_prev)
            for z_curr, z_prev in zip(z_traj, z_traj_prev)
        ]
    else:
        z_traj_extrapolated = z_traj

    coeffs_x, coeffs_y = solve_primal_step(
        z_traj_extrapolated, u_traj, segment_times, start_xy, goal_xy, v_start, v_end, rho
    )

    # Neue x-Trajektorie berechnen (aus Polynomkoeffizienten)
    x_traj = []
    for i in range(len(segment_times) - 1):
        t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], 30)
        a_x = coeffs_x[i].value
        a_y = coeffs_y[i].value
        x_vals = [evaluate_polynomial(a_x, t) for t in t_vals]
        y_vals = [evaluate_polynomial(a_y, t) for t in t_vals]
        x_traj.append(np.column_stack((x_vals, y_vals)))

    # PROJEKTION (z^{k+1})
    x_vals_list = [segment[:, 0] for segment in x_traj]
    y_vals_list = [segment[:, 1] for segment in x_traj]

    projected_x, projected_y = project_segments_to_convex_regions(
        coeffs_x, coeffs_y, segment_times, A_list, b_list
    )
    z_traj = [np.column_stack((x, y)) for x, y in zip(projected_x, projected_y)]

    # DUAL UPDATE
    u_traj = [u + (x - z) for u, x, z in zip(u_traj, x_traj, z_traj)]

    # KONVERGENZTEST
    max_diff = max(np.linalg.norm(x - z, ord=np.inf) for x, z in zip(x_traj, z_traj))
    print(f"Max segment difference: {max_diff:.5f}")

    if max_diff < eps:
        print("Konvergenz erreicht.")
        ax = pdc.visualize_environment(Al=A_list, bl=b_list, p=path_real, planar=True)

        ax.plot(start_xy[0], start_xy[1], 'go', label='Start')
        ax.plot(goal_xy[0], goal_xy[1], 'bo', label='Goal')
        ax.plot(forest[:, 0], forest[:, 1], "o", color="green", label="Bäume")

        # Projizierte Segmente (rot)
        already_labeled = set()
        for x_vals, y_vals in zip(projected_x, projected_y):
            label = "projiziert" if "projiziert" not in already_labeled else ""
            ax.plot(x_vals, y_vals, 'r-', linewidth=2, label=label)
            already_labeled.add("projiziert")

        # Aktuelle Trajektorie (bunt)
        for segment, color in zip(x_traj, colors):
            ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=1.5)

        ax.set_title(f"ADMM Iteration {k+1}")
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        plt.show()
        break

    # OPTIONAL: Zwischenstand visualisieren
    ax = pdc.visualize_environment(Al=A_list, bl=b_list, p=path_real, planar=True)

    ax.plot(start_xy[0], start_xy[1], 'go', label='Start')
    ax.plot(goal_xy[0], goal_xy[1], 'bo', label='Goal')
    ax.plot(forest[:, 0], forest[:, 1], "o", color="green", label="Bäume")

    # Projizierte Segmente (rot)
    already_labeled = set()
    for x_vals, y_vals in zip(projected_x, projected_y):
        label = "projiziert" if "projiziert" not in already_labeled else ""
        ax.plot(x_vals, y_vals, 'r-', linewidth=2, label=label)
        already_labeled.add("projiziert")

    # Aktuelle Trajektorie (bunt)
    for segment, color in zip(x_traj, colors):
        ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=1.5)

    ax.set_title(f"ADMM Iteration {k+1}")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    plt.pause(0.1)
    plt.clf()

    

    z_traj_prev = z_traj.copy()


