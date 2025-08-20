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

    reassigned_segments = []

    segment_assignments = []
    region_segment_count = [0] * len(A_list)

    for i in range(len(segment_times) - 1):
        a_x = coeffs_x[i].value
        a_y = coeffs_y[i].value

        t_vals = np.linspace(0, segment_times[i + 1] - segment_times[i], n_samples)
        x_vals = evaluate_polynomial(a_x, t_vals)
        y_vals = evaluate_polynomial(a_y, t_vals)
        segment_points = np.stack([x_vals, y_vals], axis=1)

        # Check if segment lies fully inside any region
        for j, (A, b) in enumerate(zip(A_list, b_list)):
            if is_inside_polyhedron(A, b, segment_points):
                projected_segments_x.append(x_vals)
                projected_segments_y.append(y_vals)
                segment_assignments.append(j)
                region_segment_count[j] += 1
                break
        else:
            # Projection required
            best_proj_start = None
            best_proj_end = None
            closest_dist = float('inf')
            best_region_idx = None

            start_point = segment_points[0]
            end_point = segment_points[-1]

            if use_path_guidance and path_reference is not None:
                ref_start = find_closest_point_on_path(path_reference, start_point)
                ref_end   = find_closest_point_on_path(path_reference, end_point)
            else:
                ref_start = start_point
                ref_end   = end_point

            for j, (A, b) in enumerate(zip(A_list, b_list)):
                if use_path_guidance and path_reference is not None:
                    proj_start = project_point_to_polyhedron_with_reference(
                        A, b, start_point, ref_start, lambd=lambda_path
                    )
                    proj_end   = project_point_to_polyhedron_with_reference(
                        A, b, end_point, ref_end, lambd=lambda_path
                    )
                else:
                    proj_start = project_point_to_polyhedron(A, b, start_point)
                    proj_end   = project_point_to_polyhedron(A, b, end_point)

                proj_mid = (proj_start + proj_end) / 2
                orig_mid = (start_point + end_point) / 2
                dist = np.linalg.norm(proj_mid - orig_mid)

                if dist < closest_dist:
                    closest_dist = dist
                    best_proj_start = proj_start
                    best_proj_end = proj_end
                    best_region_idx = j

            if best_proj_start is not None and best_proj_end is not None:
                x_vals_proj = np.linspace(best_proj_start[0], best_proj_end[0], len(t_vals))
                y_vals_proj = np.linspace(best_proj_start[1], best_proj_end[1], len(t_vals))
                projected_segments_x.append(x_vals_proj)
                projected_segments_y.append(y_vals_proj)
                segment_assignments.append(best_region_idx)
                region_segment_count[best_region_idx] += 1
            else:
                # fallback to unprojected segment
                projected_segments_x.append(x_vals)
                projected_segments_y.append(y_vals)
                segment_assignments.append(-1)  # Mark as unassigned

    # Postprocess: ensure each region has at least one assigned segment
    for uncovered_idx, count in enumerate(region_segment_count):
        print("Region:", uncovered_idx, "Segmente", count)
        if count == 0:
            lower_idx = uncovered_idx - 1 if uncovered_idx - 1 >= 0 else None
            upper_idx = uncovered_idx + 1 if uncovered_idx + 1 < len(A_list) else None

            candidates = []
            for neighbour in [lower_idx, upper_idx]:
                if neighbour is None:
                    continue

                seg_indices = [i for i, idx in enumerate(segment_assignments) if idx == neighbour]
                if not seg_indices:
                    continue

                # order rule: take last seg from lower neighbour, first seg from upper neighbour
                if neighbour < uncovered_idx:
                    donor_seg = max(seg_indices)
                else:
                    donor_seg = min(seg_indices)

                # projection distance
                start_point = np.array([projected_segments_x[donor_seg][0], projected_segments_y[donor_seg][0]])
                end_point   = np.array([projected_segments_x[donor_seg][-1], projected_segments_y[donor_seg][-1]])
                proj_start  = project_point_to_polyhedron(A_list[uncovered_idx], b_list[uncovered_idx], start_point)
                proj_end    = project_point_to_polyhedron(A_list[uncovered_idx], b_list[uncovered_idx], end_point)
                dist        = np.linalg.norm(((proj_start + proj_end) / 2) - ((start_point + end_point) / 2))

                candidates.append((dist, neighbour, donor_seg, proj_start, proj_end))

            if not candidates:
                print(f"WARNING: Could not find neighbour donor for region {uncovered_idx}")
                continue

            # pick best candidate
            candidates.sort(key=lambda c: c[0])
            _, donor_idx, s_idx, proj_start, proj_end = candidates[0]


            start_point = np.array([projected_segments_x[s_idx][0], projected_segments_y[s_idx][0]])
            end_point   = np.array([projected_segments_x[s_idx][-1], projected_segments_y[s_idx][-1]])

            proj_start = project_point_to_polyhedron(A_list[uncovered_idx], b_list[uncovered_idx], start_point)
            proj_end   = project_point_to_polyhedron(A_list[uncovered_idx], b_list[uncovered_idx], end_point)

            new_x = np.linspace(proj_start[0], proj_end[0], len(projected_segments_x[s_idx]))
            new_y = np.linspace(proj_start[1], proj_end[1], len(projected_segments_y[s_idx]))

            projected_segments_x[s_idx] = new_x
            projected_segments_y[s_idx] = new_y

            print(f"Reassigning segment {s_idx} from region {donor_idx} → region {uncovered_idx}")
            reassigned_segments.append((s_idx, donor_idx, uncovered_idx))

            region_segment_count[uncovered_idx] += 1
            region_segment_count[donor_idx] -= 1

    print("\nUpdated region assignments after projection fix:")
    for idx, count in enumerate(region_segment_count):
        print(f"Region {idx}: Segmente {count}")

    return projected_segments_x, projected_segments_y, reassigned_segments


