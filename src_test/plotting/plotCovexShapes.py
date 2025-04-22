from input.trees2D import generate_forest, create_occupancy_grid
from pathfinder.AStar import astar
from utils.path_manipulation import simplify_path
import pydecomp as pdc
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


def compute_polygon_from_halfspaces(A, b):
    """
    Berechnet die exakten Eckpunkte eines 2D-Polyeders, 
    definiert durch Ax <= b.
    """
    from itertools import combinations

    vertices = []
    dim = A.shape[1]

    # Alle Paare von Ungleichungen
    for (i, j) in combinations(range(len(A)), 2):
        Ai, bi = A[i], b[i]
        Aj, bj = A[j], b[j]
        
        # Lösen der Gleichungen
        A_mat = np.vstack([Ai, Aj])
        b_vec = np.array([bi, bj])

        try:
            vert = np.linalg.solve(A_mat, b_vec)
            # Prüfen, ob der Punkt alle Ungleichungen erfüllt
            if np.all(A @ vert <= b + 1e-5):
                vertices.append(vert)
        except np.linalg.LinAlgError:
            # Parallel oder singulär -> ignorieren
            continue

    vertices = np.array(vertices)
    if len(vertices) == 0:
        return None

    # Konvexe Hülle der gültigen Punkte
    hull = ConvexHull(vertices)
    return vertices[hull.vertices]



# Parameter
n_trees = 80
area_size = (25, 25)
min_distance = 2.0

# Baum- und Grid-Erzeugung
forest = generate_forest(n_trees, area_size, min_distance)
grid = create_occupancy_grid(forest, area_size, area_size, tree_size=0)

# Start- und Zielpunkt
start = (0, 0)
goal = (area_size[0] - 1, area_size[1] - 1)

# A* Pfadsuche
path = astar(grid, start, goal)

# Pfad prüfen
if path is None or len(path) == 0:
    print("Kein Pfad gefunden!")
    exit()

# Pfad Vereinfachung
path_simplified = simplify_path(path)

# Für pydecomp
path_real = np.array([(x, y) for (x, y) in path_simplified])

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
path_array = np.array([(x, y) for (x, y) in path])

# Originalpfad einzeichnen
ax.plot(path_array[:, 0], path_array[:, 1], '-', color="blue", linewidth=2, label="Originalpfad (A*)")

# Start und Ziel einzeichnen (optional, sieht gut aus)
ax.plot(start[0], start[1], "bs", markersize=10, label="Start")
ax.plot(goal[0], goal[1], "gs", markersize=10, label="Ziel")

# Legende und Plot anzeigen
ax.legend()
plt.show()
