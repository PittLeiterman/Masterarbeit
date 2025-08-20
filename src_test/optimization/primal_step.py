import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def get_snap_cost_matrix(dt):
    Q = np.zeros((6, 6))
    Q[4, 4] = 24**2 * dt
    Q[4, 5] = Q[5, 4] = 24 * 120 * dt**2 / 2
    Q[5, 5] = 120**2 * dt**3 / 3
    return Q



def minimum_snap_trajectory(start_xy, goal_xy, v_start, v_end, path, psi, num_segments=30):

    # Zeitverteilung
    distance = np.linalg.norm(np.array(goal_xy) - np.array(start_xy))
    avg_velocity = (np.linalg.norm(v_start) + np.linalg.norm(v_end)) / 2
    total_time = distance / avg_velocity if avg_velocity != 0 else 1.0
    segment_times = np.linspace(0, total_time, num_segments + 1)

    coeffs_x = []
    coeffs_y = []
    constraints = []
    cost = 0


    # Variablen und Kosten definieren
    for i in range(num_segments):
        a_x = cp.Variable(6)
        a_y = cp.Variable(6)
        coeffs_x.append(a_x)
        coeffs_y.append(a_y)

        # Snap-Kosten: Gewichtung auf höchstem Ableitungsanteil
        dt = segment_times[i+1] - segment_times[i]
        Q = get_snap_cost_matrix(dt)
        cost += cp.quad_form(a_x, Q) + cp.quad_form(a_y, Q)

        # Mittelpunkt im lokalen Zeitintervall
        t_local = (segment_times[i+1] - segment_times[i]) / 2
        T_mid = np.array([1, t_local, t_local**2, t_local**3, t_local**4, t_local**5])

        # Evaluierter Punkt auf dem Polynomsegment
        x_mid = a_x @ T_mid
        y_mid = a_y @ T_mid

        # Zielpunkt aus dem Pfad (nur der i-te Punkt, nicht i+1!)
        target_x = 0.5 * (path[i][0] + path[i+1][0])
        target_y = 0.5 * (path[i][1] + path[i+1][1])

        # Quadratischer Abstand zum Zielpunkt
        cost += psi * cp.sum_squares(x_mid - target_x)
        cost += psi * cp.sum_squares(y_mid - target_y)

    # Startbedingungen
    T0 = np.array([1, 0, 0, 0, 0, 0])
    T0_dot = np.array([0, 1, 0, 0, 0, 0])
    constraints += [
        coeffs_x[0] @ T0 == start_xy[0],
        coeffs_y[0] @ T0 == start_xy[1],
        coeffs_x[0] @ T0_dot == v_start[0],
        coeffs_y[0] @ T0_dot == v_start[1],
    ]

    # Zielbedingungen
    dt_end = segment_times[-1] - segment_times[-2]
    T1 = np.array([1, dt_end, dt_end**2, dt_end**3, dt_end**4, dt_end**5])
    T1_dot = np.array([0, 1, 2*dt_end, 3*dt_end**2, 4*dt_end**3, 5*dt_end**4])
    constraints += [
        coeffs_x[-1] @ T1 == goal_xy[0],
        coeffs_y[-1] @ T1 == goal_xy[1],
        coeffs_x[-1] @ T1_dot == v_end[0],
        coeffs_y[-1] @ T1_dot == v_end[1]
    ]

    # C³-Kontinuität
    for i in range(num_segments - 1):
        dt = segment_times[i+1] - segment_times[i]

        # Ende von Segment i
        T_end = np.array([1, dt, dt**2, dt**3, dt**4, dt**5])
        T_dot_end = np.array([0, 1, 2*dt, 3*dt**2, 4*dt**3, 5*dt**4])
        T_ddot_end = np.array([0, 0, 2, 6*dt, 12*dt**2, 20*dt**3])
        T_dddot_end = np.array([0, 0, 0, 6, 24*dt, 60*dt**2])

        # Anfang von Segment i+1 (immer t=0)
        T_start = np.array([1, 0, 0, 0, 0, 0])
        T_dot_start = np.array([0, 1, 0, 0, 0, 0])
        T_ddot_start = np.array([0, 0, 2, 0, 0, 0])
        T_dddot_start = np.array([0, 0, 0, 6, 0, 0])

        constraints += [
            coeffs_x[i] @ T_end == coeffs_x[i+1] @ T_start,
            coeffs_y[i] @ T_end == coeffs_y[i+1] @ T_start,

            coeffs_x[i] @ T_dot_end == coeffs_x[i+1] @ T_dot_start,
            coeffs_y[i] @ T_dot_end == coeffs_y[i+1] @ T_dot_start,

            coeffs_x[i] @ T_ddot_end == coeffs_x[i+1] @ T_ddot_start,
            coeffs_y[i] @ T_ddot_end == coeffs_y[i+1] @ T_ddot_start,

            coeffs_x[i] @ T_dddot_end == coeffs_x[i+1] @ T_dddot_start,
            coeffs_y[i] @ T_dddot_end == coeffs_y[i+1] @ T_dddot_start,
        ]

    # Optimierung lösen
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    print("Status:", prob.status)
    print("Kostenwert:", prob.value)

    return coeffs_x, coeffs_y, segment_times



def evaluate_polynomial(coeffs, t):
    return sum(c * t**i for i, c in enumerate(coeffs))


def solve_primal_step(z_traj, u_traj, segment_times, start_xy, goal_xy, v_start, v_end, rho, psi, path):
    import cvxpy as cp

    num_segments = len(segment_times) - 1
    coeffs_x = []
    coeffs_y = []
    constraints = []
    cost = 0

    for i in range(num_segments):
        a_x = cp.Variable(6)
        a_y = cp.Variable(6)
        coeffs_x.append(a_x)
        coeffs_y.append(a_y)

        dt = segment_times[i+1] - segment_times[i]
        Q = get_snap_cost_matrix(dt)
        cost += cp.quad_form(a_x, Q) + cp.quad_form(a_y, Q)

        # Abtastung zur "Zielverfolgung"
        t_vals = np.linspace(0, dt, len(z_traj[i]))
        for j, t in enumerate(t_vals):
            T = np.array([t**k for k in range(6)])
            target_x = z_traj[i][j] - u_traj[i][j]
            cost += (rho / 2) * cp.square(a_x @ T - target_x[0])
            cost += (rho / 2) * cp.square(a_y @ T - target_x[1])

            t_local = (segment_times[i+1] - segment_times[i]) / 2
            T_mid = np.array([1, t_local, t_local**2, t_local**3, t_local**4, t_local**5])

            # Evaluierter Punkt auf dem Polynomsegment
            x_mid = a_x @ T_mid
            y_mid = a_y @ T_mid

            # Zielpunkt aus dem Pfad (nur der i-te Punkt, nicht i+1!)
            target_x = 0.5 * (path[i][0] + path[i+1][0])
            target_y = 0.5 * (path[i][1] + path[i+1][1])

            # Quadratischer Abstand zum Zielpunkt
            cost += psi * cp.sum_squares(x_mid - target_x)
            cost += psi * cp.sum_squares(y_mid - target_y)

    # Startbedingungen
    T0 = np.array([1, 0, 0, 0, 0, 0])
    T0_dot = np.array([0, 1, 0, 0, 0, 0])
    constraints += [
        coeffs_x[0] @ T0 == start_xy[0],
        coeffs_y[0] @ T0 == start_xy[1],
        coeffs_x[0] @ T0_dot == v_start[0],
        coeffs_y[0] @ T0_dot == v_start[1]
    ]

    # Zielbedingungen
    dt_end = segment_times[-1] - segment_times[-2]
    T1 = np.array([1, dt_end, dt_end**2, dt_end**3, dt_end**4, dt_end**5])
    T1_dot = np.array([0, 1, 2*dt_end, 3*dt_end**2, 4*dt_end**3, 5*dt_end**4])
    constraints += [
        coeffs_x[-1] @ T1 == goal_xy[0],
        coeffs_y[-1] @ T1 == goal_xy[1],
        coeffs_x[-1] @ T1_dot == v_end[0],
        coeffs_y[-1] @ T1_dot == v_end[1]
    ]

    # C³-Kontinuität
    for i in range(num_segments - 1):
        dt = segment_times[i+1] - segment_times[i]

        T_end     = np.array([1, dt, dt**2, dt**3, dt**4, dt**5])
        T_dot     = np.array([0, 1, 2*dt, 3*dt**2, 4*dt**3, 5*dt**4])
        T_ddot    = np.array([0, 0, 2, 6*dt, 12*dt**2, 20*dt**3])
        T_dddot   = np.array([0, 0, 0, 6, 24*dt, 60*dt**2])

        T_start   = np.array([1, 0, 0, 0, 0, 0])
        T_dot0    = np.array([0, 1, 0, 0, 0, 0])
        T_ddot0   = np.array([0, 0, 2, 0, 0, 0])
        T_dddot0  = np.array([0, 0, 0, 6, 0, 0])

        constraints += [
            coeffs_x[i] @ T_end == coeffs_x[i+1] @ T_start,
            coeffs_y[i] @ T_end == coeffs_y[i+1] @ T_start,
            coeffs_x[i] @ T_dot == coeffs_x[i+1] @ T_dot0,
            coeffs_y[i] @ T_dot == coeffs_y[i+1] @ T_dot0,
            coeffs_x[i] @ T_ddot == coeffs_x[i+1] @ T_ddot0,
            coeffs_y[i] @ T_ddot == coeffs_y[i+1] @ T_ddot0,
            coeffs_x[i] @ T_dddot == coeffs_x[i+1] @ T_dddot0,
            coeffs_y[i] @ T_dddot == coeffs_y[i+1] @ T_dddot0,
        ]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    return coeffs_x, coeffs_y
