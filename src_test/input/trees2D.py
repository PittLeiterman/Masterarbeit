import numpy as np

border_trees = [
    # Untere Kante (y=0)
    (0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0), (8.0, 0.0),
    (10.0, 0.0), (12.0, 0.0), (14.0, 0.0), (16.0, 0.0), (18.0, 0.0),
    (20.0, 0.0), (22.0, 0.0), (24.0, 0.0), (25.0, 0.0),

    # Obere Kante (y=25)
    (0.0, 25.0), (2.0, 25.0), (4.0, 25.0), (6.0, 25.0), (8.0, 25.0),
    (10.0, 25.0), (12.0, 25.0), (14.0, 25.0), (16.0, 25.0), (18.0, 25.0),
    (20.0, 25.0), (22.0, 25.0), (24.0, 25.0), (25.0, 25.0),

    # Linke Kante (x=0), ohne Ecken
    (0.0, 2.0), (0.0, 4.0), (0.0, 6.0), (0.0, 8.0), (0.0, 10.0),
    (0.0, 12.0), (0.0, 14.0), (0.0, 16.0), (0.0, 18.0), (0.0, 20.0),
    (0.0, 22.0), (0.0, 24.0),

    # Rechte Kante (x=25), ohne Ecken
    (25.0, 2.0), (25.0, 4.0), (25.0, 6.0), (25.0, 8.0), (25.0, 10.0),
    (25.0, 12.0), (25.0, 14.0), (25.0, 16.0), (25.0, 18.0), (25.0, 20.0),
    (25.0, 22.0), (25.0, 24.0)
]

def generate_forest(n_trees, area_size, min_distance):
    np.random.seed(0)  # 👈 sets the random seed
    
    trees = []
    while len(trees) < n_trees:
        x = np.random.uniform(0, area_size[0])
        y = np.random.uniform(0, area_size[1])
        new_tree = np.array([x, y])
        
        if all(np.linalg.norm(new_tree - t) >= min_distance for t in trees):
            trees.append(new_tree)
    
    return np.array(trees)

def load_forest_from_file(filepath):
    """
    Lädt Baumpositionen aus einer Textdatei.
    Jede Zeile: x y (getrennt durch Leerzeichen)
    """

    
    try:
        forest = np.loadtxt(filepath)
        if forest.ndim == 1:
            forest = np.expand_dims(forest, axis=0)  # handle single-line file
        return np.vstack([forest, np.array(border_trees)])
    except Exception as e:
        raise RuntimeError(f"Fehler beim Laden der Datei '{filepath}': {e}")

def create_occupancy_grid(forest, area_size, grid_size, tree_size=0):
    """
    Erstellt ein 2D-Occupancy-Grid, bei dem Bäume eine Fläche von mehreren Zellen belegen können.

    forest: (n,2) array der Baumpositionen
    area_size: (width, height) in Metern
    grid_size: (rows, cols) des Gitters
    tree_size: int, Radius in Zellen (0 = nur Mittelpunkt, 1 = 3x3, 2 = 5x5 usw.)
    """
    width, height = area_size
    rows, cols = grid_size
    grid = np.zeros((rows, cols), dtype=np.uint8)
    
    for tree in forest:
        x, y = tree
        col = int((x / width) * cols)
        row = int((y / height) * rows)
        
        for i in range(-tree_size, tree_size + 1):
            for j in range(-tree_size, tree_size + 1):
                r = row + i
                c = col + j
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = 1
    
    return grid

