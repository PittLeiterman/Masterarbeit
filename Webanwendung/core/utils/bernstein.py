# utils/bernstein.py
import numpy as np
from math import comb
import scipy.linalg

def _power_to_bernstein_matrix(n: int) -> np.ndarray:
    T = np.zeros((n+1, n+1))
    for k in range(n+1):
        for m in range(n+1):
            T[k, m] = comb(k, m) / comb(n, m) if k >= m else 0.0
    return T

def _monomial_to_bernstein_matrix(n: int, dt: float) -> np.ndarray:
    T = _power_to_bernstein_matrix(n)        # (n+1) x (n+1)
    D = np.diag([dt**m for m in range(n+1)]) # skaliert u=t/dt
    return T @ D                              # (n+1) x (n+1)

def build_T_block(segment_times, degree: int = 5) -> np.ndarray:
    blocks = []
    for i in range(len(segment_times)-1):
        dt = float(segment_times[i+1] - segment_times[i])
        Ti = _monomial_to_bernstein_matrix(degree, dt)  # (n+1)x(n+1) => 6x6
        blocks.append(Ti)
    return scipy.linalg.block_diag(*blocks)  # (6S x 6S)
