import numpy as np
from scipy.spatial import cKDTree

def evaluate_polynomial(coeffs, t_vals):
    return np.polyval(coeffs[::-1], t_vals)

def is_inside_polyhedron(A, b, points, tol=1e-6):
    return np.all(A @ points.T <= b[:, None] + tol, axis=0).all()


def _is_feasible(A, b, x, tol=1e-10):
    return np.all(A @ x <= b + tol)

def _project_to_halfspace_boundary(c, a, beta):
    # Project c onto the line a^T x = beta
    denom = np.dot(a, a)
    if denom == 0:
        return None
    t = (np.dot(a, c) - beta) / denom
    return c - t * a

def _solve_equalities(a1, b1, a2, b2, tol=1e-14):
    # Solve [a1^T; a2^T] x = [b1; b2]
    M = np.vstack([a1, a2])
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    if abs(det) < tol:
        return None  # parallel or nearly singular
    return np.linalg.solve(M, np.array([b1, b2], dtype=float))

import numpy as np

def project_point_to_polyhedron(A, b, point, tol=1e-10):
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)
    p = np.asarray(point, float).reshape(2)
    if A.shape[1] != 2:
        raise ValueError("2D only (A is m x 2).")

    # If already feasible
    if np.all(A @ p <= b + tol):
        return p.copy()

    m = A.shape[0]
    cand = []

    # (1) Feasible line-feet for all constraints (vectorized)
    An2 = np.einsum("ij,ij->i", A, A)
    denom_ok = An2 > tol
    if np.any(denom_ok):
        t = (A @ p - b) / An2
        X = p - t[:, None] * A
        feas = np.all(A @ X.T <= b[:, None] + tol, axis=0)
        cand.append(X[denom_ok & feas])

    # (2) All feasible pairwise intersections (vertices), vectorized
    if m >= 2:
        I, J = np.triu_indices(m, 1)
        ai, aj = A[I], A[J]
        bi, bj = b[I], b[J]
        det = ai[:, 0]*aj[:, 1] - ai[:, 1]*aj[:, 0]
        mask = np.abs(det) > tol
        if np.any(mask):
            ai, aj, bi, bj, det = ai[mask], aj[mask], bi[mask], bj[mask], det[mask]
            Xv = np.empty((det.shape[0], 2))
            Xv[:, 0] = (aj[:, 1]*bi - ai[:, 1]*bj) / det
            Xv[:, 1] = (-aj[:, 0]*bi + ai[:, 0]*bj) / det
            feas = np.all(A @ Xv.T <= b[:, None] + tol, axis=0)
            cand.append(Xv[feas])

    if not cand:
        return p.copy()
    C = np.vstack([c for c in cand if c.size]) if len(cand) > 1 else cand[0]
    if C.size == 0:
        return p.copy()
    d2 = np.sum((C - p) ** 2, axis=1)
    return C[np.argmin(d2)]

def project_point_to_polyhedron_with_reference(A, b, point, ref, lambd=1.0, tol=1e-10):
    if lambd < 0:
        raise ValueError("lambd must be >= 0")
    p = np.asarray(point, float).reshape(2)
    r = np.asarray(ref, float).reshape(2)
    c = (p + lambd * r) / (1.0 + lambd)
    return project_point_to_polyhedron(A, b, c, tol=tol)





def find_closest_point_on_path(path_points, query_point):
    dists = np.linalg.norm(path_points - query_point, axis=1)
    return path_points[np.argmin(dists)]


def project_segments_to_convex_regions(
    coeffs_x,
    coeffs_y,
    segment_times,
    A_list,
    b_list,
    n_samples=30
):
    projected_segments_x = []
    projected_segments_y = []

    reassigned_segments = []

    segment_assignments = []
    region_segment_count = [0] * len(A_list)

    # Precompute region centroids (approximate, using least-squares pseudo-inverse)
    region_centroids = [np.mean(np.linalg.pinv(A) @ b, axis=0) for A, b in zip(A_list, b_list)]

    for i in range(len(segment_times) - 1):
        a_x = coeffs_x[i].value
        a_y = coeffs_y[i].value

        t_vals = np.linspace(0, segment_times[i + 1] - segment_times[i], n_samples)
        x_vals = evaluate_polynomial(a_x, t_vals)
        y_vals = evaluate_polynomial(a_y, t_vals)
        segment_points = np.stack([x_vals, y_vals], axis=1)

        # Check if segment lies fully inside any region
        assigned = False
        for j, (A, b) in enumerate(zip(A_list, b_list)):
            if is_inside_polyhedron(A, b, segment_points):
                if not assigned:
                    # Store the segment once
                    projected_segments_x.append(x_vals)
                    projected_segments_y.append(y_vals)
                    segment_assignments.append([j])   # list of memberships
                    assigned = True
                else:
                    # Add extra membership
                    segment_assignments[-1].append(j)
                region_segment_count[j] += 1

        if not assigned:
            # Projection required
            best_proj_start = None
            best_proj_end = None
            closest_dist = float('inf')
            best_region_idx = None

            start_point = segment_points[0]
            end_point = segment_points[-1]

            ref_start = start_point
            ref_end   = end_point

            # Only check closest few regions instead of all
            mid = (start_point + end_point) / 2
            close_regions = np.argsort([np.linalg.norm(c - mid) for c in region_centroids])[:3]

            for j in close_regions:
                A, b = A_list[j], b_list[j]
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
                segment_assignments.append([best_region_idx])
                region_segment_count[best_region_idx] += 1
            else:
                # fallback to unprojected segment
                projected_segments_x.append(x_vals)
                projected_segments_y.append(y_vals)
                segment_assignments.append([-1])  # Mark as unassigned


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


