import numpy as np
import math
import warnings
from scipy.spatial.transform import Rotation as R
# http://192.168.1.32:8883
ex = np.array([1,0,0])
ey = np.array([0,1,0])
ez = np.array([0,0,1])
l0 = 0.061
l1 = 0.0435
l2 = 0.08285
l3 = 0.08285
l4 = 0.07385
l5 = 0.05457
L1 = l0+l1
L4 = l4+l5
q = np.full((5, 4), np.nan)
def hat(k: np.ndarray) -> np.ndarray:
    """Skew-symmetric (hat) operator for a 3D vector k."""
    kx, ky, kz = k
    return np.array([[0, -kz, ky],
                     [kz, 0, -kx],
                     [-ky, kx, 0]], dtype=float)

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v.copy()
    return v / n

def rot(k: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation: rotate by angle theta (radians) about unit axis k."""
    khat = hat(unit(k))
    I = np.eye(3)
    return I + np.sin(theta) * khat + (1 - np.cos(theta)) * (khat @ khat)

# ---------- canonical subproblems ----------
def subprob1(k: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Find theta such that R(k, theta) @ p1 = p2.
    Assumes p1 and p2 have equal norms and both are perpendicular to k (or use their projected parts).
    """
    k = unit(k)
    p1p = p1 - k * (k @ p1)
    p2p = p2 - k * (k @ p2)
    n1 = np.linalg.norm(p1p); n2 = np.linalg.norm(p2p)
    if n1 == 0 or n2 == 0:
        return 0.0
    p1p = p1p / n1
    p2p = p2p / n2
    c = np.clip(p1p @ p2p, -1.0, 1.0)
    s = k @ (np.cross(p1p, p2p))
    return np.arctan2(s, c)

def subprob3(k: np.ndarray, p1: np.ndarray, p2: np.ndarray, d: float) -> np.ndarray:
    """
    Find theta such that || p2 - R(k, theta) @ p1 || = d.
    Returns up to two solutions as a 1D numpy array of thetas (radians).
    """
    k = unit(k)
    p1p = p1 - k * (k @ p1)
    p2p = p2 - k * (k @ p2)
    a = np.linalg.norm(p1p)
    b = np.linalg.norm(p2p)
    eps = 1e-12
    if a < eps or b < eps:
        if abs(np.linalg.norm(p2 - p1) - d) < 1e-9:
            return np.array([0.0])
        return np.array([])
    c_val = (a*a + b*b - d*d) / (2*a*b)
    if c_val < -1.0 - 1e-12 or c_val > 1.0 + 1e-12:
        return np.array([])
    c_val = np.clip(c_val, -1.0, 1.0)
    gamma = np.arccos(c_val)
    theta0 = subprob1(k, p1p / a, p2p / b)
    return np.array([theta0 + gamma, theta0 - gamma])

def subprob4(k: np.ndarray, h: np.ndarray, p: np.ndarray, d: float) -> np.ndarray:
    """
    Find theta such that k^T R(h, theta) p = d.
    Returns up to two solutions.
    """
    k = unit(k)
    h = unit(h)
    # Expand k^T R(h,θ) p = (u^T p) cosθ + (v^T p) sinθ + (a * h^T p)
    a = h @ k
    u = k - a * h           # component of k orthogonal to h
    v = np.cross(h, k)      # also orthogonal to h, and orthogonal to u

    A = u @ p
    B = v @ p
    C = a * (h @ p)

    E = d - C
    r = np.hypot(A, B)      # sqrt(A^2 + B^2)

    eps = 1e-12
    if r < eps:
        # Then A ~ B ~ 0 -> equation reduces to C = d. If satisfied, infinite solutions; return 0.
        if abs(E) < 1e-9:
            return np.array([0.0])
        return np.array([])

    # Solve A cosθ + B sinθ = E  =>  cos(θ - φ) = E / r
    phi = np.arctan2(B, A)
    arg = E / r
    if arg < -1.0 - 1e-12 or arg > 1.0 + 1e-12:
        return np.array([])
    arg = np.clip(arg, -1.0, 1.0)
    alpha = np.arccos(arg)
    return np.array([phi + alpha, phi - alpha])

def invkin_subproblems_Dofbot(Rot: np.ndarray, Pot: np.ndarray) -> np.ndarray:
    q = np.full((5, 4), np.nan)

    # Subproblem 4 -> theta = q2+q3+q4
    k = -ey
    h = ez
    p = ex
    d = ez.T @ (Rot @ ex)
    thetatmp = subprob4(k, h, p, d)
    if len(thetatmp) == 1:
        theta = [thetatmp[0], np.nan, thetatmp[0], np.nan]
    else:
        theta = [thetatmp[0], thetatmp[1], thetatmp[0], thetatmp[1]]

    # Subproblem 1 -> q1
    for ii in range(4):
        if not np.isnan(theta[ii]):
            k = ez
            p1 = rot(-ey, theta[ii]) @ ex
            p2 = Rot @ ex
            q[0, ii] = subprob1(k, p1, p2)
            print(q[0,ii])

    # Subproblem 1 -> q5
    for ii in range(4):
        if not np.isnan(theta[ii]):
            k = ex
            p1 = rot(ey, theta[ii]) @ ez
            p2 = Rot.T @ ez
            q[4, ii] = subprob1(k, p1, p2)

    # Subproblem 3  -> q3
    for ii in range(2):
        if not np.isnan(theta[ii]):
            Pprime = rot(ez, -q[0, ii]) @ (Pot - L1 * ez - rot(ey, -theta[ii]) @ (L4 * ex))
            k = -ey
            p1 = l3 * ez
            p2 = l2 * ex
            d = np.linalg.norm(Pprime)
            q3tmp = subprob3(k, p1, p2, d)
            if q3tmp.size == 1:
                q[2, ii] = q3tmp[0]
            elif q3tmp.size >= 2:
                q[2, ii] = q3tmp[0]
                q[2, ii + 2] = q3tmp[1]

    # Subproblem 1 -> q2
    for ii in range(4):
        if not np.isnan(q[2, ii]):
            Pprime = rot(ez, -q[0, ii]) @ (Pot - L1 * ez - rot(ey, -theta[ii]) @ (L4 * ex))
            k = -ey
            p1 = l2 * ex - (rot(-ey, q[2, ii]) @ (l3 * ez))
            p2 = Pprime
            q[1, ii] = subprob1(k, p1, p2)
    # Solving for q4
    for ii in range(4):
        if not np.isnan(q[2, ii]):
            q[3, ii] = theta[ii] - q[1, ii] - q[2, ii]

    # ---- remove nonvalid columns ----
    cols = []
    for ii in range(4):
        if not np.isnan(q[:, ii]).any():
            cols.append(q[:, ii])
    if len(cols) == 0:
        return np.empty((0, 5))
    q_valid = np.stack(cols, axis=1)
    q_deg = np.rad2deg(q_valid)

    for i in range(q_deg.shape[0]):
        for j in range(q_deg.shape[1]):
            if q_deg[i, j] < -180.0:
                q_deg[i, j] += 360.0
            elif q_deg[i, j] > 360.0:
                q_deg[i, j] -= 360.0
    return q_deg.T

import numpy as np
import time
from Arm_Lib import Arm_Device
Arm = Arm_Device()
time.sleep(.2)
Rot_I = np.eye(3)  # Identity rotation
Pot_sample = np.array([0.16030867, 0.13451494, 0.03215322])
# Compute IK
q_solutions = invkin_subproblems_Dofbot(Rot_I, Pot_sample)
if q_solutions.shape[0] == 0:
    print("No valid IK solution found!")
else:
    for idx, q in enumerate(q_solutions):
        q = q.astype(int)
        q = np.append(q, 0)
        print(f"\nSolution {idx+1} of {q_solutions.shape[0]}: {q}")

        user_input = input("Press Enter to use this solution or type 'q' to abort: ").lower()
        if user_input == 'q':
            print("Aborted")
            continue
        else:
            print(f"Selected solution {idx+1}")
            Arm.Arm_serial_servo_write6(q[0]+90,q[1]+90,q[2]+90,q[3]+90,q[4]+90,q[5]+90, 500)
            time.sleep(0.5)
            for i in range(5):
                print(i+1, Arm.Arm_serial_servo_read(i+1))