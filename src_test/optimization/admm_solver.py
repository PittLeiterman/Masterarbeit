import numpy as np

def compute_Q(N, D, t):
    dim = 3  # 3D trajectories
    Q = np.zeros((dim*N*(D+1), dim*N*(D+1)))

    for k in range(N):
        delta_t = t[k+1] - t[k]
        for j in range(4, D+1):
            for m in range(4, D+1):
                coeff = (j*(j-1)*(j-2)*(j-3)) * (m*(m-1)*(m-2)*(m-3))
                integral = (delta_t ** (j + m - 7)) / (j + m - 7)
                q_block = coeff * integral * np.eye(3)

                row = k*(D+1) + j
                col = k*(D+1) + m
                Q[3*row:3*row+3, 3*col:3*col+3] = q_block
    return Q


def compute_S(K, t_sample, N, D, t_segments):
    dim = 3
    S = np.zeros((dim*K, dim*N*(D+1)))
    
    for l, tl in enumerate(t_sample):
        for k in range(N):
            if t_segments[k] <= tl <= t_segments[k+1]:
                for j in range(D+1):
                    value = (tl - t_segments[k]) ** j
                    block = value * np.eye(3)
                    idx_row = 3 * l
                    idx_col = 3 * (k*(D+1) + j)
                    S[idx_row:idx_row+3, idx_col:idx_col+3] = block
                break
    return S

def project_C(u):
    # Assume C = union of convex polytopes C_i
    # Here, for illustration, pick the projection onto the closest polytope
    # Each C_i is represented as a function: Pi(u) = projection onto C_i
    
    projections = [Pi(u) for Pi in list_of_projections]
    distances = [np.linalg.norm(u - zi) for zi in projections]
    i_star = np.argmin(distances)
    return projections[i_star]



def admm_solver(Q, S, project_C, rho=1.0, max_iter=100):
    n = Q.shape[0]
    m = S.shape[0]
    
    alpha = np.zeros(n)
    z = np.zeros(m)
    lam = np.zeros(m)
    
    Q_rho = Q + rho * S.T @ S
    inv_Q_rho = np.linalg.inv(Q_rho)
    
    for _ in range(max_iter):
        # α-update
        alpha = inv_Q_rho @ (rho * S.T @ z - S.T @ lam)
        
        # z-update via projection onto union of convex sets
        u = S @ alpha + (1 / rho) * lam
        z = project_C(u)
        
        # λ-update
        lam = lam + rho * (S @ alpha - z)
    
    return alpha
