import numpy as np
from scipy.spatial import distance

def evaluate_polynomial(coeffs, t_vals):
    return np.array([sum(c * t**i for i, c in enumerate(coeffs)) for t in t_vals])

def is_inside_polyhedron(A, b, points, tol=1e-6):
    # Prüfe, ob alle Punkte Ax <= b erfüllen
    return np.all(A @ points.T <= b[:, None] + tol)

def project_point_to_polyhedron(A, b, point):
    # Projektionsproblem: min ||x - point||^2 s.t. A x <= b
    import cvxpy as cp
    x = cp.Variable(2)
    objective = cp.Minimize(cp.sum_squares(x - point))
    constraints = [A @ x <= np.array(b).flatten()]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return x.value if x.value is not None else point

def project_segments_to_convex_regions(coeffs_x, coeffs_y, segment_times, A_list, b_list, n_samples=30):
    projected_segments_x = []
    projected_segments_y = []

    for i in range(len(segment_times) - 1):
        a_x = coeffs_x[i].value
        a_y = coeffs_y[i].value

        t_vals = np.linspace(0, segment_times[i+1] - segment_times[i], n_samples)
        x_vals = evaluate_polynomial(a_x, t_vals)
        y_vals = evaluate_polynomial(a_y, t_vals)
        segment_points = np.stack([x_vals, y_vals], axis=1)

        # Prüfe, ob das Segment vollständig in einer Region liegt
        in_region = False
        for A, b in zip(A_list, b_list):
            if is_inside_polyhedron(A, b, segment_points):
                in_region = True
                break

        if in_region:
            # Keine Änderung
            projected_segments_x.append(x_vals)
            projected_segments_y.append(y_vals)
        else:
            # Projektiere alle Punkte einzeln (alternativ: mittleren Punkt nehmen)
            closest = None
            closest_dist = float('inf')

            for A, b in zip(A_list, b_list):
                midpoint = np.mean(segment_points, axis=0)
                proj = project_point_to_polyhedron(A, b, midpoint)
                dist = np.linalg.norm(midpoint - proj)
                if dist < closest_dist:
                    closest = proj
                    closest_dist = dist

            # Verschiebe das ganze Segment so, dass der Mittelpunkt in die Region fällt
            translation = closest - np.mean(segment_points, axis=0)
            x_vals_proj = x_vals + translation[0]
            y_vals_proj = y_vals + translation[1]
            projected_segments_x.append(x_vals_proj)
            projected_segments_y.append(y_vals_proj)

    return projected_segments_x, projected_segments_y
