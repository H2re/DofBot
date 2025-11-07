import numpy as np
from scipy.spatial.transform import Rotation as R
from qpsolvers import solve_qp
import general_robotics_toolbox as rox

def defineDofbot():
    l0 = 61e-3
    l1 = 43.5e-3
    l2 = 82.85e-3
    l3 = 82.85e-3
    l4 = 73.85e-3
    l5 = 54.57e-3
    
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])
    
    P01 = (l0 + l1) * ez
    P12 = np.zeros(3)
    P23 = l2 * ex
    P34 = -l3 * ez
    P45 = np.zeros(3)
    P5T = -(l4 + l5) * ex
    
    H = np.array([ez, -ey, -ey, -ey, -ex]).T
    P = np.array([P01, P12, P23, P34, P45, P5T]).T
    joint_type = [0, 0, 0, 0, 0]
    robot = rox.Robot(H, P, joint_type)
    # Set joint limits (in radians)
    robot.joint_lower_limit = np.deg2rad(np.array([0,0,0,0,0]))
    robot.joint_upper_limit = np.deg2rad(np.array([180,180,180,180,270]))
    return robot

import numpy as np
from scipy.spatial.transform import Rotation as R
from qpsolvers import solve_qp
import general_robotics_toolbox as rox

def defineDofbot():
    l0 = 61e-3
    l1 = 43.5e-3
    l2 = 82.85e-3
    l3 = 82.85e-3
    l4 = 73.85e-3
    l5 = 54.57e-3
    
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])
    
    P01 = (l0 + l1) * ez
    P12 = np.zeros(3)
    P23 = l2 * ex
    P34 = -l3 * ez
    P45 = np.zeros(3)
    P5T = -(l4 + l5) * ex
    
    H = np.array([ez, -ey, -ey, -ey, -ex]).T
    P = np.array([P01, P12, P23, P34, P45, P5T]).T
    joint_type = [0, 0, 0, 0, 0]
    robot = rox.Robot(H, P, joint_type)
    # Set joint limits (in radians)
    robot.joint_lower_limit = np.deg2rad(np.array([0,0,0,0,0]))
    robot.joint_upper_limit = np.deg2rad(np.array([180,180,180,180,270]))
    return robot

def qpPathGen(robot, q0, P0Td, R0Td, epsilon_r, epsilon_p, q_prime_min, q_prime_max, N):
    n = len(q0)
    lambda_vals = np.linspace(0, 1, N + 1)
    # Get initial forward kinematics using rox
    T0 = rox.fwdkin(robot, q0)
    R0T0 = T0.R
    P0T0 = T0.p
    # Compute Path in Task Space
    ER0 = R0T0 @ R0Td.T
    # Convert rotation matrix to axis-angle using rox
    k_temp, theta0 = rox.R2rot(ER0)
    k_hat = np.array(k_temp).flatten()

    Euldes_lambda = np.zeros((3, len(lambda_vals)))
    Pdes_lambda = np.zeros((3, len(lambda_vals)))

    dP0T_dlambda = P0Td - P0T0
    der_dlambda = np.zeros(len(lambda_vals))

    for k in range(len(lambda_vals)):
        theta = (1 - lambda_vals[k]) * theta0
        Rot = rox.rot(k_hat, (1 - lambda_vals[k]) * theta0)
        Euldes_lambda[:, k] = np.flip(R.from_matrix(Rot).as_euler('zyx'))
        Pdes_lambda[:, k] = (1 - lambda_vals[k]) * P0T0 + lambda_vals[k] * P0Td
        der_dlambda[k] = -theta0
    # Solve QP Problem and Generate Joint Space Path
    q_prime = np.zeros((n, len(lambda_vals)))
    q_lambda = np.zeros((n, len(lambda_vals) + 1))
    q_lambda[:, 0] = q0
    exitflag = np.zeros(len(lambda_vals), dtype=int)
    P0T_lambda = np.zeros((3, len(lambda_vals) + 1))
    R0T_lambda = np.zeros((3, 3, len(lambda_vals) + 1))
    P0T_lambda[:, 0] = P0T0
    R0T_lambda[:, :, 0] = R0T0
    Eul_lambda = np.zeros((3, len(lambda_vals) + 1))
    Eul_lambda[:, 0] = np.flip(R.from_matrix(R0T_lambda[:, :, 0]).as_euler('zyx'))
    qprev = q0.copy()
    
    for k in range(len(lambda_vals)):
        lb, ub = qprimelimits_full(robot.joint_lower_limit, robot.joint_upper_limit, qprev, N, q_prime_max, q_prime_min)
        J = rox.robotjacobian(robot, qprev)
        vt = dP0T_dlambda
        vr = der_dlambda[k] * k_hat

        H = getqp_H(qprev, J, vr, vt, epsilon_r, epsilon_p)
        f = getqp_f(qprev, epsilon_r, epsilon_p)
        
        H = (H + H.T) / 2
        H += 1e-8 * np.eye(H.shape[0])
        # Solve QP using qpsolvers
        q_prime_temp = solve_qp(H, f, lb=lb, ub=ub, solver='quadprog')
        exitflag[k] = 1
        q_prime_temp = q_prime_temp[:n]
        q_prime[:, k] = q_prime_temp
        qprev = qprev + (1 / N) * q_prime_temp
        qprev = np.clip(qprev, robot.joint_lower_limit, robot.joint_upper_limit)
        q_lambda[:, k + 1] = qprev
        Ttemp = rox.fwdkin(robot, qprev)
        Rtemp = Ttemp.R
        Ptemp = Ttemp.p
        
        P0T_lambda[:, k + 1] = Ptemp
        R0T_lambda[:, :, k + 1] = Rtemp
        Eul_lambda[:, k + 1] = np.flip(R.from_matrix(Rtemp).as_euler('zyx'))
    
    # Chop off excess
    q_lambda = q_lambda[:, :-1]
    P0T_lambda = P0T_lambda[:, :-1]
    R0T_lambda = R0T_lambda[:, :, :-1]
    return q_lambda, lambda_vals, P0T_lambda, R0T_lambda


def qprimelimits_full(joint_lower_limit, joint_upper_limit, qprev, N, qpmax, qpmin):
    n = len(joint_lower_limit)
    # Compute limits due to joint stops
    lb_js = N * (joint_lower_limit - qprev)
    ub_js = N * (joint_upper_limit - qprev)
    # Compare and find most restrictive bound
    lb = np.zeros(n + 2)
    ub = np.zeros(n + 2)
    ub[-2:] = 1
    lb[:n] = np.maximum(lb_js, qpmin)
    ub[:n] = np.minimum(ub_js, qpmax)
    return lb, ub

def getqp_H(dq, J, vr, vp, er, ep):
    n = len(dq)
    vr = np.atleast_2d(vr).reshape(-1, 1)
    vp = np.atleast_2d(vp).reshape(-1, 1)
    J_aug = np.hstack([J, np.zeros((6, 2))])
    V_mat = np.vstack([
        np.hstack([np.zeros((3, n)), vr, np.zeros((3, 1))]),
        np.hstack([np.zeros((3, n)), np.zeros((3, 1)), vp])
    ])
    E_mat = np.vstack([
        np.hstack([np.zeros((1, n)), [[np.sqrt(er)]], [[0]]]),
        np.hstack([np.zeros((1, n)), [[0]], [[np.sqrt(ep)]]])
    ])
    H1 = J_aug.T @ J_aug
    H2 = V_mat.T @ V_mat
    H3 = -2 * J_aug.T @ V_mat
    H3 = (H3 + H3.T) / 2
    H4 = E_mat.T @ E_mat
    return 2 * (H1 + H2 + H3 + H4)

def getqp_f(dq, er, ep):
    f = -2 * np.hstack([np.zeros(len(dq)), [er, ep]])
    return f


def main():
    robot = defineDofbot()
    print("\nRobot created successfully")
    # Initial joint configuration
    q0 = np.deg2rad(np.array([90, 90, 90, 90, 90]))
    print(f"\nInitial joint angles (deg): {np.rad2deg(q0)}")
    T0 = rox.fwdkin(robot, q0)
    print(T0)

    qd = np.deg2rad(np.array([45, 45, 45, 45, 45]))
    H_des = rox.fwdkin(robot, qd)
    sol = rox.iterative_invkin(robot, H_des, q0)
    print(f"\nDesired:")
    print(np.rad2deg(sol[1]))
    # QP parameters
    epsilon_r = 0.1
    epsilon_p = 0.1
    # Joint velocity limits (rad/s equivalent)
    q_prime_min = -np.inf * np.ones(5)
    q_prime_max = np.inf * np.ones(5)
    N = 100

    print(f"\nGenerating path with N={N} segments...")
    q_lambda, lambda_vals, P0T_lambda, R0T_lambda = qpPathGen(
        robot, q0, H_des.p, H_des.R, 
        epsilon_r, epsilon_p, 
        q_prime_min, q_prime_max, N
    )
    
    if q_lambda is not None:
        print("\nPath generation successful!")
        print(f"Number of waypoints: {q_lambda.shape[1]}")
        print(f"\nFinal joint angles (deg): {np.rad2deg(q_lambda[:, -1])}")
        print(f"Final position: {P0T_lambda[:, -1]}")
        print(f"Position error: {np.linalg.norm(P0T_lambda[:, -1] - H_des.p):.6f} m")
    else:
        print("\nPath generation FAILED!")


if __name__ == "__main__":
    main()