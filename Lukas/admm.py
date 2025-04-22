# %%

import numpy as np
import cvxpy as cp

# ----- Utility functions for setting up problems, testing against cvxpy ----- #ß


# Class to store problem data representing the following problem
# min_x 1/2 x'Hx + g'x
# subject to: l <= Ax <= u
class QPProb:
    def __init__(self, H, g, A, l, u):
        self.H, self.g, self.A, self.l, self.u = H, g, A, l, u
        self.nx, self.nc = H.shape[0], A.shape[0]

        if len(self.g.shape) == 1:
            self.g = self.g.reshape((self.nx, 1))
        if len(self.l.shape) == 1:
            self.l = self.l.reshape((self.nc, 1))
        if len(self.u.shape) == 1:
            self.u = self.u.reshape((self.nc, 1))

    def primal_res(self, x, z):
        return np.linalg.norm(self.A @ x - z, np.inf)

    def dual_res(self, x, lamb):
        return np.linalg.norm(self.H @ x + self.g + self.A.T @ lamb, np.inf)


# Generate a random qp with equality and inequality constraints
def rand_qp(nx, n_eq, n_ineq):
    H = np.random.randn(nx, nx)
    H = H.T @ H + np.eye(nx)
    H = H + H.T

    A = np.random.randn(n_eq, nx)
    C = np.random.randn(n_ineq, nx)

    active_ineq = np.random.randn(n_ineq) > 0.5

    mu = np.random.randn(n_eq)
    lamb = (np.random.randn(n_ineq)) * active_ineq

    x = np.random.randn(nx)
    b = A @ x
    d = C @ x - np.random.randn(n_ineq) * (~active_ineq)

    g = -H @ x - A.T @ mu - C.T @ lamb

    x = cp.Variable(nx)
    prob = cp.Problem(
        cp.Minimize((1 / 2) * cp.quad_form(x, np.array(H)) + g.T @ x),
        [A @ x == b, C @ x >= d],
    )
    prob.solve()

    return QPProb(
        H,
        g,
        np.vstack((A, C)),
        np.concatenate((b, d)),
        np.concatenate((b, np.full(n_ineq, np.inf))),
    ), x.value.reshape((nx, 1))


# Solve a QPProb using cvxpy
def get_sol(prob):
    x = cp.Variable(prob.nx)
    cp_prob = cp.Problem(
        cp.Minimize((1 / 2) * cp.quad_form(x, np.array(prob.H)) + prob.g.T @ x),
        [prob.A @ x >= prob.l[:, 0], prob.A @ x <= prob.u[:, 0]],
    )
    cp_prob.solve()
    return x.value.reshape((prob.nx, 1))


# Generate A and B for a random controllable linear system that is underactuated
def random_linear_system(nx, nu):
    assert nx % 2 == 0

    A = np.random.randn(nx, nx)
    U, S, Vt = np.linalg.svd(A)
    eigs = 3 * np.random.rand(nx) - 1.5
    A = U @ np.diagflat(eigs) @ Vt
    A = (A + A.T) / 2

    B = np.random.randn(nx, nu)

    assert check_controllability(A, B)

    return A, B


# Check the controllability of a linear system
def check_controllability(A, B):
    nx = A.shape[0]
    P = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(nx)])
    return np.linalg.matrix_rank(P) == nx and np.linalg.cond(P) < 1e10


# Setup a basic mpc qp with the given horizon and unitary quadratic cost
# The primal variables are the stacked state and controls, i.e. [u_1, x_1, u_2, x_2, ...]
def setup_mpc_qp(Ad, Bd, N):
    nx, nu = Ad.shape[0], Bd.shape[1]

    # Cost matrix, identity hessian, no cost gradient
    H = np.diagflat(np.ones((nx + nu) * N))
    g = np.zeros(((nx + nu) * N, 1))

    # Equality constraints for the dynamics, essentially x_next = Ax + Bu
    A_eq = np.zeros((nx * N, (nx + nu) * N))
    for k in range(N):
        if k == 0:
            A_eq[0:nx, 0 : nx + nu] = np.hstack([Bd, -np.identity(nx)])
        else:
            A_eq[nx * k : nx * (k + 1), (nx + nu) * k - nx : (nx + nu) * (k + 1)] = (
                np.hstack([Ad, Bd, -np.identity(nx)])
            )

    # Artificial torque constraints for u (selects out u_1, u_2, etc from the stacked variables)
    A_ineq = np.kron(np.identity(N), np.hstack([np.identity(nu), np.zeros((nu, nx))]))

    # Bounds, zeros for the eq constraints, and then torque limits
    l = np.vstack([np.zeros((nx * N, 1)), -1 * np.ones((A_ineq.shape[0], 1))])
    u = np.vstack([np.zeros((nx * N, 1)), 1 * np.ones((A_ineq.shape[0], 1))])

    return QPProb(H, g, np.vstack([A_eq, A_ineq]), l, u)


# ------------------ Solver functions for implementing ADMM ------------------ #ß


# ADMM iteration which solves the following problem:
# min_x 1/2 x'Hx + g'x
# subject to: Ax = z
#             l <= z <= u
# where lamb are the dual variables
def admm_iter(prob, x, z, lamb, rho, debug=False):
    # Form matrix
    schur_mat = prob.H + 1e-6 * np.identity(prob.nx) + rho * prob.A.T @ prob.A

    # Update x
    kkt_lhs = -prob.g + 1e-6 * x + prob.A.T @ (rho * z - lamb)
    if np.linalg.norm(kkt_lhs, np.inf) < 1e-100:  # Fails below 1e-162
        x = x * 0
    else:
        # x = pcg.solve(schur_mat, kkt_lhs)
        x = np.linalg.solve(schur_mat, kkt_lhs)

    # Update z
    z = np.clip(prob.A @ x + 1 / rho * lamb, prob.l, prob.u)

    # Update lamb
    lamb = lamb + rho * (prob.A @ x - z)

    # Calculate residuals
    primal_res = prob.primal_res(x, z)
    dual_res = prob.dual_res(x, lamb)

    # Update rho
    if primal_res > 10 * dual_res:
        rho = 2 * rho
    elif dual_res > 10 * primal_res:
        rho = 1 / 2 * rho
    rho = np.clip(rho, 1e-6, 1e6)

    # Output
    if debug:
        print("r_p: %2.2e\tr_d: %2.2e" % (primal_res, dual_res))

    return x, z, lamb, rho


# Outer loop for ADMM which checks the primal and dual residuals for convergence
def admm_solve(prob, tol=1e-3, max_iter=1000, debug=False):
    x = np.zeros((prob.nx, 1))
    z = np.zeros((prob.nc, 1))
    lamb = np.zeros((prob.nc, 1))
    rho = 0.1

    for iter in range(max_iter):
        if debug:
            print("Iter %d\tRho: %1.2e\t" % (iter + 1, rho), end="")
        x, z, lamb, rho = admm_iter(prob, x, z, lamb, rho, debug=debug)
        primal_res = prob.primal_res(x, z)
        dual_res = prob.dual_res(x, lamb)
        if primal_res < tol and dual_res < tol:
            print("Finished in %d iters" % (iter + 1))
            break

    return x, z, lamb


# ----------------------------------- Test ----------------------------------- #ß
nx, nu, N = 10, 2, 10
A, B = random_linear_system(nx, nu)
prob = setup_mpc_qp(A, B, N)

x0 = 10 * np.random.rand(nx)
prob.l[0:nx, 0] = -A @ x0
prob.u[0:nx, 0] = -A @ x0

x_sol = get_sol(prob)

x, z, lamb = admm_solve(prob, debug=False, tol=1e-3, max_iter=4000)
print(np.linalg.norm(x - x_sol, np.inf))  # Should match around 1e-1/1e-2 if tol = 1e-3
u = np.kron(np.identity(N), np.hstack([np.identity(nu), np.zeros((nu, nx))])) @ x
print(u)


# Small problem test
H = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
g = np.array([[-8], [-3], [-3]])
A = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
l = np.array([[3], [0], [-10], [-10], [-10]])
u = np.array([[3], [0], [np.inf], [np.inf], [np.inf]])
prob = QPProb(H, g, A, l, u)

prob, x_sol = rand_qp(10, 2, 2)

x, z, lamb = admm_solve(prob, debug=False, tol=1e-15)
print(np.linalg.norm(x - x_sol, np.inf))
