import numpy as np

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

def project_point_to_polyhedron_with_reference(A, b, point, ref, lambd=1.0):
    import cvxpy as cp
    x = cp.Variable(2)
    objective = cp.Minimize(cp.sum_squares(x - point) + lambd * cp.sum_squares(x - ref))
    constraints = [A @ x <= np.array(b).flatten()]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return x.value if x.value is not None else point


def find_closest_point_on_path(path_points, query_point):
    dists = np.linalg.norm(path_points - query_point, axis=1)
    return path_points[np.argmin(dists)]


def project_segments_to_convex_regions(
    coeffs_x,
    coeffs_y,
    segment_times,
    A_list,
    b_list,
    path_reference=None,
    use_path_guidance=False,
    lambda_path=1.0,
    n_samples=30
):
    projected_segments_x = []
    projected_segments_y = []

    for i in range(len(segment_times) - 1):
        a_x = coeffs_x[i].value
        a_y = coeffs_y[i].value

        t_vals = np.linspace(0, segment_times[i + 1] - segment_times[i], n_samples)
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
            # Projektion notwendig
            best_proj_start = None
            best_proj_end = None
            closest_dist = float('inf')

            start_point = segment_points[0]
            end_point = segment_points[-1]

            if use_path_guidance and path_reference is not None:
                ref_start = find_closest_point_on_path(path_reference, start_point)
                ref_end = find_closest_point_on_path(path_reference, end_point)
            else:
                ref_start = start_point
                ref_end = end_point

            for A, b in zip(A_list, b_list):
                if use_path_guidance and path_reference is not None:
                    proj_start = project_point_to_polyhedron_with_reference(A, b, start_point, ref_start, lambd=lambda_path)
                    proj_end = project_point_to_polyhedron_with_reference(A, b, end_point, ref_end, lambd=lambda_path)
                else:
                    proj_start = project_point_to_polyhedron(A, b, start_point)
                    proj_end = project_point_to_polyhedron(A, b, end_point)

                proj_mid = (proj_start + proj_end) / 2
                orig_mid = (start_point + end_point) / 2
                dist = np.linalg.norm(proj_mid - orig_mid)

                if dist < closest_dist:
                    closest_dist = dist
                    best_proj_start = proj_start
                    best_proj_end = proj_end

            if best_proj_start is not None and best_proj_end is not None:
                x_vals_proj = np.linspace(best_proj_start[0], best_proj_end[0], len(t_vals))
                y_vals_proj = np.linspace(best_proj_start[1], best_proj_end[1], len(t_vals))
                projected_segments_x.append(x_vals_proj)
                projected_segments_y.append(y_vals_proj)
            else:
                projected_segments_x.append(x_vals)
                projected_segments_y.append(y_vals)

    return projected_segments_x, projected_segments_y
