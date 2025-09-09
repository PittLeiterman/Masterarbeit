# optimization/sampling.py
import numpy as np
import scipy.sparse as sp
from scipy.sparse import block_diag as sp_block_diag, csr_matrix

def build_Phi(segment_times, m_per_seg: int):
    """Block-diagonal design matrix for monomial basis in *t* (not normalized): Φ_i[k,:] = [1,t,t^2..t^5]."""
    S = len(segment_times) - 1
    blocks = []
    for i in range(S):
        dt = segment_times[i+1] - segment_times[i]
        t_vals = np.linspace(0.0, dt, m_per_seg)
        Phi_i = np.vstack([t_vals**k for k in range(6)]).T # (m x 6)
        blocks.append(csr_matrix(Phi_i))
    return sp_block_diag(blocks, format="csr")             # ((S*m) x (6S))
