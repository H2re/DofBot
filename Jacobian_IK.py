import math
import numpy as np
import general_robotics_toolbox as rox

def jacobian_inverse(robot, q0, Rd, Pd, Nmax, tol, lambda_d=0.01):
    n = len(q0)
    q = np.zeros((n, Nmax+1))
    q[:, 0] = q0
    p0T = np.zeros((3, 1))
    RPY0T = np.zeros((3, Nmax+1))
    iternum = 0

    # Initial FK
    H = rox.fwdkin(robot, q[:, 0])
    R = H.R
    P = np.array([[H.p[0]], [H.p[1]], [H.p[2]]])

    dR = R @ Rd.T
    r = np.array(rox.R2rpy(dR))[None]
    dX = np.concatenate((r.T, P - Pd))

    while (np.abs(dX) > tol).any() and iternum < Nmax:
        H = rox.fwdkin(robot, q[:, iternum])
        R = H.R
        p0T = np.array([[H.p[0]], [H.p[1]], [H.p[2]]])

        dR = R @ Rd.T
        r = np.array(rox.R2rpy(dR))[None]
        dX = np.concatenate((r.T, p0T - Pd))

        Jq = rox.robotjacobian(robot, q[:, iternum])

        # --- Damped Least Squares update ---
        JJ = Jq.T @ Jq + lambda_d**2 * np.eye(n)
        dq = np.linalg.solve(JJ, Jq.T @ dX)
        q[:, iternum+1] = q[:, iternum] - dq.flatten()

        iternum += 1

    return q[:, iternum]

def main():
    q0 = np.deg2rad([90, 90, 90, 90, 90])
    tol = np.array([0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001])
    Nmax = 200
    l0 = 61e-3; l1 = 43.5e-3; l2 = 82.85e-3
    l3 = 82.85e-3; l4 = 73.85e-3; l5 = 54.57e-3

    ex = np.array([1,0,0])
    ey = np.array([0,1,0])
    ez = np.array([0,0,1])

    P01 = (l0+l1)*ez
    P12 = np.zeros(3)
    P23 = l2*ex
    P34 = -l3*ez
    P45 = np.zeros(3)
    P5T = -(l4+l5)*ex

    H_axes = np.array([ez, -ey, -ey, -ey, -ex]).T
    P_vectors = np.array([P01, P12, P23, P34, P45, P5T]).T
    joint_type = [0,0,0,0,0]

    robot = rox.Robot(H_axes, P_vectors, joint_type)

    # Desired pose
    qd = np.deg2rad([100, 100, 100, 100, 100])
    H_des = rox.fwdkin(robot, qd)
    print(H_des)
    Rd = H_des.R
    Pd = np.array([[H_des.p[0]], [H_des.p[1]], [H_des.p[2]]])

    # IK
    q_sol = jacobian_inverse(robot, q0, Rd, Pd, Nmax, tol)
    q_deg = np.rad2deg(q_sol)
    print(q_deg)

if __name__ == "__main__":
    main()