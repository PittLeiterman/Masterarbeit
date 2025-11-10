import numpy as np

border_trees = []

for x in range(0, 26):
    border_trees.append((x, 0.0))

for x in range(0, 26):
    border_trees.append((x, 25.0))

for y in range(1, 25):
    border_trees.append((0.0, y))

for y in range(1, 25):
    border_trees.append((25.0, y))


def generate_forest(n_trees, area_size, min_distance):
    np.random.seed(0)
    
    trees = []
    while len(trees) < n_trees:
        x = np.random.uniform(0, area_size[0])
        y = np.random.uniform(0, area_size[1])
        new_tree = np.array([x, y])
        
        if all(np.linalg.norm(new_tree - t) >= min_distance for t in trees):
            trees.append(new_tree)
    
    return np.array(trees)

def load_forest_from_file(filepath, include_border=True):
    try:
        forest = np.loadtxt(filepath)
        if forest.ndim == 1:
            forest = np.expand_dims(forest, axis=0)
        if include_border:
            return np.vstack([forest, np.array(border_trees)])
        else:
            return forest
    except Exception as e:
        raise RuntimeError(f"Fehler beim Laden der Datei '{filepath}': {e}")


def create_occupancy_grid(forest, area_size, grid_size, tree_size=0):
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

