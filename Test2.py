import numpy as np
import cvxpy as cp

def rot(k_hat, theta):
    """Rodrigues rotation formula"""
    if np.linalg.norm(k_hat) < 1e-8:
        return np.eye(3)
    k = k_hat / np.linalg.norm(k_hat)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
    return R

def qp_path_gen(robot, q0, P0Td, R0Td, epsilon_r, epsilon_p, q_prime_min, q_prime_max, N):
    """
    Generate joint-space path for robot using QP
    robot: object with methods fwdkin(q) -> (R,P) and jacobian(q) -> 6xn
    q0: initial joint angles (n,)
    P0Td, R0Td: desired end-effector pose
    epsilon_r, epsilon_p: weights for rotation/position
    q_prime_min, q_prime_max: incremental joint limits
    N: number of steps
    """
    n = len(q0)
    lambda_arr = np.linspace(0,1,N+1)

    # Initial FK
    R0T0, P0T0 = robot.fwdkin(q0)

    # Rotation error
    ER0 = R0T0 @ R0Td.T
    theta0 = np.arccos((np.trace(ER0)-1)/2)
    if np.abs(theta0) < 1e-6:
        k_hat = np.zeros(3)
    else:
        k_hat = (1/(2*np.sin(theta0))) * np.array([ER0[2,1]-ER0[1,2],
                                                   ER0[0,2]-ER0[2,0],
                                                   ER0[1,0]-ER0[0,1]])

    # Position and rotation increments
    dP0T_dlambda = P0Td - P0T0
    der_dlambda = -theta0 * np.ones(N+1)

    # Initialize outputs
    q_lambda = np.zeros((n,N+1))
    q_lambda[:,0] = q0
    P0T_lambda = np.zeros((3,N+1))
    R0T_lambda = np.zeros((3,3,N+1))
    P0T_lambda[:,0] = P0T0
    R0T_lambda[:,:,0] = R0T0

    q_prev = q0.copy()

    for k in range(N+1):
        # Compute Jacobian
        J = robot.jacobian(q_prev)

        # Task-space velocities
        vt = dP0T_dlambda
        vr = der_dlambda[k]*k_hat

        # QP variables
        dq = cp.Variable(n)

        # QP cost function: ||J*dq - [vr; vt]||^2 + regularization
        A = np.vstack([J[0:3,:], J[3:6,:]])  # split for rotation/position
        b = np.hstack([vr, vt])
        cost = cp.quad_form(A@dq - b, np.eye(len(b))) + epsilon_r*cp.sum_squares(dq) + epsilon_p*cp.sum_squares(dq)

        # Bounds
        lb = q_prime_min
        ub = q_prime_max
        constraints = [dq >= lb, dq <= ub]

        # Solve QP
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)

        if dq.value is None:
            print("QP failed at step", k)
            break

        # Update joint angles
        q_prev = q_prev + dq.value / N
        q_lambda[:,k] = q_prev

        # Update FK
        Rtemp, Ptemp = robot.fwdkin(q_prev)
        P0T_lambda[:,k] = Ptemp
        R0T_lambda[:,:,k] = Rtemp

    return q_lambda, lambda_arr, P0T_lambda, R0T_lambda