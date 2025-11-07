import math
import numpy as np
import general_robotics_toolbox as rox

def main():
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

    H = np.array([ez, -ey, -ey, -ey, -ex]).T   # shape (3,5)
    P = np.array([P01, P12, P23, P34, P45, P5T]).T   # shape (3,6)
    joint_type = [0, 0, 0, 0, 0]
    robot = rox.Robot(H, P, joint_type)
    q0 = np.deg2rad(np.array([90, 90, 90, 90, 90]))
    qd = np.deg2rad(np.array([90, 90, 90, 90, 90]))
    H_des = rox.fwdkin(robot, qd)
    print(H_des)
    solutions = rox.iterative_invkin(robot, H_des, q0)
    print("\nInverse Kinematics Solution (degrees):")
    print(np.rad2deg(solutions[1]))

if __name__ == "__main__":
    main()