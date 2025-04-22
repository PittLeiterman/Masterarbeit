import numpy as np

def generate_forest(n_trees, area_size, min_distance):
    trees = []
    
    while len(trees) < n_trees:
        x = np.random.uniform(0, area_size[0])
        y = np.random.uniform(0, area_size[1])
        new_tree = np.array([x, y])
        
        if all(np.linalg.norm(new_tree - t) >= min_distance for t in trees):
            trees.append(new_tree)
    
    return np.array(trees)

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

