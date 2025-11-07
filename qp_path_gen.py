import numpy as np

try:
    from cvxopt import matrix, solvers
    CVXOPT_AVAILABLE = True
    solvers.options['show_progress'] = False
except Exception:
    CVXOPT_AVAILABLE = False

from scipy.optimize import minimize, LinearConstraint, Bounds

# Rotation / conversion helpers
def rot(axis, theta):
    """Rodrigues rotation matrix from axis (3,) and angle theta (scalar)."""
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(3)
    k = axis / norm
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
    return R

def vrrotmat2vec(R):
    """
    Convert rotation matrix R to axis-angle representation (axis (3,), angle)
    Output matches MATLAB vrrotmat2vec: returns (x,y,z,theta) with axis unit vector.
    """
    R = np.asarray(R, dtype=float)
    
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if np.isclose(theta, 0.0):
        
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    if np.isclose(theta, np.pi):
        
        Rx = np.sqrt((R[0,0] + 1) / 2.0)
        Ry = np.sqrt((R[1,1] + 1) / 2.0)
        Rz = np.sqrt((R[2,2] + 1) / 2.0)
        axis = np.array([Rx, Ry, Rz])
        axis = axis / np.linalg.norm(axis)
        return np.concatenate((axis, [theta]))
    
    denom = 2*np.sin(theta)
    kx = (R[2,1] - R[1,2]) / denom
    ky = (R[0,2] - R[2,0]) / denom
    kz = (R[1,0] - R[0,1]) / denom
    axis = np.array([kx, ky, kz])
    axis = axis / np.linalg.norm(axis)
    return np.concatenate((axis, [theta]))

def rotm2eul(R):
    """
    ZYX Euler angles (phi_z, theta_y, psi_x) similar to MATLAB rotm2eul with default 'ZYX'.
    Returns [z, y, x] (i.e., [yaw, pitch, roll]).
    """
    R = np.asarray(R, dtype=float)
    
    if abs(R[2,0]) < 1 - 1e-8:
        theta_y = -np.arcsin(R[2,0])
        cos_ty = np.cos(theta_y)
        psi_x = np.arctan2(R[2,1]/cos_ty, R[2,2]/cos_ty)
        phi_z = np.arctan2(R[1,0]/cos_ty, R[0,0]/cos_ty)
    else:
        
        if R[2,0] <= -1:
            theta_y = np.pi/2
            psi_x = 0.0
            phi_z = np.arctan2(R[0,1], R[0,2])
        else:
            theta_y = -np.pi/2
            psi_x = 0.0
            phi_z = np.arctan2(-R[0,1], -R[0,2])
    return np.array([phi_z, theta_y, psi_x])

# QP building blocks (H and f)
def getqp_H(dq, J, vr, vp, er, ep):
    """
    Build H matrix as in MATLAB getqp_H.
    dq : current joint vector (n,)
    J  : 6 x n jacobian
    vr : 3-vector
    vp : 3-vector
    er, ep : scalars
    Returns H: (n+2) x (n+2)
    """
    n = len(dq)
    J = np.asarray(J, dtype=float)
    
    A = np.hstack((J, np.zeros((6,2))))
    B = np.vstack(( np.hstack((np.zeros((3,n)), vr.reshape(3,1), np.zeros((3,1)) )),
                    np.hstack((np.zeros((3,n)), np.zeros((3,1)), vp.reshape(3,1))) ))   # 6 x (n+2)
    
    H1 = A.T @ A
    
    H2 = B.T @ B
    
    H3 = -2 * (A.T @ B)
    H3 = 0.5 * (H3 + H3.T)
    
    row1 = np.concatenate((np.zeros(n), [np.sqrt(er), 0.0]))
    row2 = np.concatenate((np.zeros(n), [0.0, np.sqrt(ep)]))
    R = np.vstack((row1, row2))
    H4 = R.T @ R
    H = 2.0 * (H1 + H2 + H3 + H4)
    
    H = 0.5 * (H + H.T)
    return H

def getqp_f(dq, er, ep):
    """Return f vector as in MATLAB getqp_f"""
    n = len(dq)
    f = -2.0 * np.concatenate((np.zeros(n), [er, ep]))
    return f.reshape(-1,)

# qprime limits
def qprimelimits_full(qlimit, qprev, N, qpmax, qpmin):
    """
    qlimit: (n,2) array: qlimit[:,0] min, qlimit[:,1] max
    qprev: (n,)
    qpmax, qpmin: arrays length n
    returns lb, ub arrays of length n+2 (last two correspond to the additional variables)
    """
    qlimit = np.asarray(qlimit, dtype=float)
    qprev = np.asarray(qprev, dtype=float)
    n = qlimit.shape[0]
    lb_js = N * (qlimit[:,0] - qprev)
    ub_js = N * (qlimit[:,1] - qprev)
    lb = np.zeros(n+2, dtype=float)
    ub = np.zeros(n+2, dtype=float)
    ub[-2] = 1.0
    ub[-1] = 1.0
    for k in range(n):
        lb[k] = lb_js[k] if lb_js[k] > qpmin[k] else qpmin[k]
        ub[k] = ub_js[k] if ub_js[k] < qpmax[k] else qpmax[k]
    return lb, ub

# QP solve wrapper
def solve_qp(H, f, lb=None, ub=None):
    """
    Solve QP: minimize 0.5 x.T H x + f.T x
    subject to lb <= x <= ub (both optional)
    Returns x (1d numpy) and success flag (True/False)
    Uses cvxopt if available otherwise scipy.optimize minimize as fallback.
    """
    H = np.asarray(H, dtype=float)
    f = np.asarray(f, dtype=float)
    n = H.shape[0]

    
    eps = 1e-9
    H_reg = H + eps * np.eye(n)

    if CVXOPT_AVAILABLE:
        P = matrix(H_reg)
        q = matrix(f.reshape(-1,1))
        
        G_list = []
        h_list = []
        if ub is not None:
            G_list.append(matrix(np.eye(n)))
            h_list.append(matrix(ub.reshape(-1,1)))
        if lb is not None:
            G_list.append(matrix(-np.eye(n)))
            h_list.append(matrix((-lb).reshape(-1,1)))
        if len(G_list) > 0:
            G = matrix(np.vstack([np.array(g).astype(float) for g in G_list]))
            h = matrix(np.vstack([np.array(hv).astype(float) for hv in h_list]))
        else:
            G = None
            h = None
        try:
            sol = solvers.qp(P, q, G, h) if G is not None else solvers.qp(P, q)
            x = np.array(sol['x']).reshape(-1)
            return x, True
        except Exception:
            return np.zeros(n), False
    else:
        
        bounds = None
        if lb is not None and ub is not None:
            bounds = Bounds(lb, ub)
        elif lb is not None:
            bounds = Bounds(lb, np.full(n, np.inf))
        elif ub is not None:
            bounds = Bounds(np.full(n, -np.inf), ub)

        def obj(x):
            return 0.5 * x @ H_reg @ x + f @ x
        def jac(x):
            return H_reg @ x + f

        x0 = np.zeros(n)
        res = minimize(obj, x0, method='trust-constr', jac=jac,
                       bounds=bounds,
                       options={'gtol':1e-8, 'xtol':1e-8, 'maxiter':200})
        return (res.x, res.success)

# Main function port
def qp_path_gen(robot, q0, P0Td, R0Td,
                epsilon_r, epsilon_p, q_prime_min, q_prime_max, N):
    """
    Ported from MATLAB qpPathGen.

    Inputs:
        robot: object with fwdkin(q), jacobian(q), and qlimit (n x 2 array)
        q0: initial joint vector (n,)
        P0Td: desired final position (3,)
        R0Td: desired final rotation (3x3)
        epsilon_r, epsilon_p: scalars (tolerances in cost)
        q_prime_min, q_prime_max: arrays length n (bounds on qprime)
        N: number of segments (integer)
    Outputs:
        q_lambda: n x N array (joint path for lambda = 0..1 step 1/N)
        lambda_vec: array of length N (0..1-1/N)
        P0T_lambda: 3 x N array of forward kinematics positions
        R0T_lambda: 3 x 3 x N array of rotation matrices
    """
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    n = len(q0)
    
    lambda_vec = np.linspace(0.0, 1.0, N+1)
    options = {}

    R0T0, P0T0 = robot.fwdkin(q0)
    R0T0 = np.asarray(R0T0, dtype=float)
    P0T0 = np.asarray(P0T0, dtype=float).reshape(3)

    # Path in task space
    ER0 = R0T0 @ (R0Td.T)
    temp = vrrotmat2vec(ER0)
    k_hat = temp[0:3]
    theta0 = temp[3]
    Euldes_lambda = np.zeros((3, len(lambda_vec)))
    Pdes_lambda = np.zeros((3, len(lambda_vec)))
    dP0T_dlambda = (P0Td - P0T0)
    der_dlambda = np.zeros(len(lambda_vec))
    for k_idx, lam in enumerate(lambda_vec):
        theta = (1.0 - lam) * theta0
        R = rot(k_hat, theta)
        eul = rotm2eul(R)
        
        Euldes_lambda[:, k_idx] = eul[::-1]
        
        Pdes_lambda[:, k_idx] = (1.0 - lam) * P0T0 + lam * dP0T_dlambda
        der_dlambda[k_idx] = -theta0

    q_prime = np.zeros((n, len(lambda_vec)))
    q_lambda = np.zeros((n, len(lambda_vec)))
    q_lambda[:, 0] = q0
    exitflag = np.zeros(len(lambda_vec), dtype=int)
    P0T_lambda = np.zeros((3, len(lambda_vec)))
    R0T_lambda = np.zeros((3, 3, len(lambda_vec)))
    P0T_lambda[:, 0] = P0T0
    R0T_lambda[:, :, 0] = R0T0
    Eul_lambda = np.zeros((3, len(lambda_vec)))
    Eul_lambda[:, 0] = rotm2eul(R0T0)[::-1]

    qprev = q0.copy()
    for k_idx, lam in enumerate(lambda_vec):
        lb, ub = qprimelimits_full(robot.qlimit, qprev, N,
                                   np.asarray(q_prime_max).reshape(-1),
                                   np.asarray(q_prime_min).reshape(-1))
        
        J = robot.jacobian(qprev)
        J = np.asarray(J, dtype=float)
        vt = dP0T_dlambda
        vr = der_dlambda[k_idx] * k_hat
        H = getqp_H(qprev, J, vr, vt, epsilon_r, epsilon_p)
        f = getqp_f(qprev, epsilon_r, epsilon_p)
        
        x, success = solve_qp(H, f, lb=lb, ub=ub)
        if not success:
            raise RuntimeError(f"QP solver failed at lambda step {k_idx}, returning early.")
        q_prime_temp = x[:n]
        exitflag[k_idx] = 1 if success else -1
        q_prime[:, k_idx] = q_prime_temp
        
        qprev = qprev + (1.0 / N) * q_prime_temp
        
        if k_idx + 1 < q_lambda.shape[1]:
            q_lambda[:, k_idx + 1] = qprev
            Rtemp, Ptemp = robot.fwdkin(qprev)
            R0T_lambda[:, :, k_idx + 1] = np.asarray(Rtemp, dtype=float)
            P0T_lambda[:, k_idx + 1] = np.asarray(Ptemp, dtype=float).reshape(3)
            Eul_lambda[:, k_idx + 1] = rotm2eul(Rtemp)[::-1]

    q_lambda = q_lambda[:, :-1]
    P0T_lambda = P0T_lambda[:, :-1]
    R0T_lambda = R0T_lambda[:, :, :-1]
    lambda_vec = lambda_vec[:-1]

    return q_lambda, lambda_vec, P0T_lambda, R0T_lambda

# Example 'robot' adapter for testing (toy)
class DummyRobot:
    def __init__(self, qlimit):
        
        self.qlimit = np.asarray(qlimit, dtype=float)
    def fwdkin(self, q):
        
        q = np.asarray(q).reshape(-1)
        P = np.zeros(3)
        P[:min(3,len(q))] = q[:min(3,len(q))]
        R = np.eye(3)
        return R, P
    def jacobian(self, q):
        n = len(q)
        
        J = np.zeros((6, n))
        for i in range(min(3, n)):
            J[i, i] = 1.0
        return J

# Only run example if module executed directly
if __name__ == "__main__":
    
    n = 6
    q0 = np.zeros(n)
    qlimit = np.vstack(( -np.ones(n)*np.pi, np.ones(n)*np.pi )).T
    robot = DummyRobot(qlimit)
    P0Td = np.array([0.1, 0.2, 0.3])
    R0Td = np.eye(3)
    q_lambda, lam, P_path, R_path = qp_path_gen(robot, q0, P0Td, R0Td,
                                                epsilon_r=1e-3, epsilon_p=1e-3,
                                                q_prime_min=-0.5*np.ones(n),
                                                q_prime_max=0.5*np.ones(n),
                                                N=10)
    print("q_lambda shape:", q_lambda.shape)
    print("lambda:", lam)