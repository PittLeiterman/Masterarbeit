import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def evaluate_polynomial(coeffs, t):
    return sum(c * t**i for i, c in enumerate(coeffs))