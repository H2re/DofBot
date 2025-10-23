import numpy as np
import math
import warnings
from scipy.spatial.transform import Rotation as R
# http://192.168.1.32:8883
ex = np.array([1,0,0])
ey = np.array([0,1,0])
ez = np.array([0,0,1])
I3 = np.array([1,0,0], [0,1,0], [0,0,1])
l0 = 0.061
l1 = 0.0435
l2 = 0.08285
l3 = 0.08285
l4 = 0.07385
l5 = 0.05457
L1 = l0+l1
L4 = l4+l5
def ik_Dofbot (POT, ROT):
# Subproblem 4 -> theta = q2+q3+q4
    k = -ey
    h = ez
    p = ex
    d = ez.T@ROT@ex
    thetatmp = subproblem4(k,h,p,d)
    if len(thetatmp)==1:
        theta = [thetatmp[0], 0, thetatmp[0], 0]
    else:
        theta = [thetatmp[0], thetatmp[1], thetatmp[0], thetatmp[1]]
    # Subproblem 1 -> q1
    for i in range(3):
        if not np.isnan(theta[i]):
            k = ez
            p1 = roty(-theta[i])@ex
            p2 = ROT@ex
            q[0,i] = subproblem1(k,p1,p2)
    # Subproblem 1 -> q5
    for i in range(3):
        if not np.isnan(theta[i]):
            k = ex
            p1 = roty(theta[i]) @ ez
            p2 = Rot.T @ ez
            q[4,i] = subproblem1(k,p1,p2)
    # Subproblem 3  -> q3
    for i in range(1):
        if not np.isnan(theta[i]):
            Pprime = rotz(-q[0, i]) @ (Pot - L1 * ez) + roty(-theta[i]) @ (L4 * ex)
            k = -ey
            p1 = l3*ez
            p2 = l2*ex
            d = np.linalg.norm(Pprime)
            q3tmp = subproblem3(k,p1,p2,d)
        if len(q3tmp)==1:
            q(3,i) = q3tmp
        else:
            q(3,i) = q3tmp(0)
            q(3,i+2) = q3tmp(1)
    # Subproblem 1 -> q2
    for i in range(3):
        if not np.isnan(q[2,i]):
            k = -ey
            p1 = l2 * ex - roty(-q[2, i]) @ (l3 * ez)
            p2 = Pprime
            q[1,i] = subproblem1(k,p1,p2)
    # Solving for q4
    for i in range(3):
        if not np.isnan(q[3,i]):
            q[3,i] = theta[i]-q[1,i]-q[2,i]
    # Remove invalid 
    q = q[:, ~np.isnan(q).any(axis=0)]
    q = np.degrees(q)
    q[q < -180] += 360
    q[q > 360] -= 360
    return q
"""
Uses code from rpiRobotics/general_robotics_toolbox_py
Copyright (c) 2018, RPI & Wason Technology LLC
"""

def hat(k):
    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]    
    return khat

if __name__ == "__main__":
    Pot = np.array([-0.0456, 0, 0.0217])
    Rot = np.eye(3)
    q = np.full((5, 4), np.nan)
    q = ik_Dofbot(Pot, Rot)
    ik_Dofbot(Pot, Rot)

def rotx(theta):
    if isinstance(theta,np.ndarray):
        theta = theta[0]
    Rx = R.from_euler('x',theta , degrees = True )
    return Rx

def roty(theta):
    if isinstance(theta,np.ndarray):
        theta = theta[0]
    Ry = R.from_euler('y',theta , degrees = True )
    return Ry

def rotz(theta):
    if isinstance(theta,np.ndarray):
        theta = theta[0]
    Rz = R.from_euler('z',theta, degrees = True )
    return Rz

def invhat(khat):
    return np.array([(-khat[1,2] + khat[2,1]),(khat[0,2] - khat[2,0]),(-khat[0,1]+khat[1,0])])/2

def subproblem0(p, q, k):
    eps = np.finfo(np.float64).eps    
    assert (np.dot(k,p) < eps) and (np.dot(k,q) < eps), \
           "k must be perpendicular to p and q"
    norm = np.linalg.norm
    ep = p / norm(p)
    eq = q / norm(q)
    theta = 2 * np.arctan2( norm(ep - eq), norm (ep + eq))
    if (np.dot(k,np.cross(p , q)) < 0):
        return -theta 
    return theta

def subproblem1(p, q, k):
    eps = np.finfo(np.float64).eps
    norm = np.linalg.norm
    if norm (np.subtract(p, q)) < np.sqrt(eps):
        return 0.0
    k = np.divide(k,norm(k))
    pp = np.subtract(p,np.dot(p, k)*k)
    qp = np.subtract(q,np.dot(q, k)*k)
    epp = np.divide(pp, norm(pp))    
    eqp = np.divide(qp, norm(qp))
    theta = subproblem0(epp, eqp, k)
    if (np.abs(norm(p) - norm(q)) > norm(p)*1e-2):
        warnings.warn("||p|| and ||q|| must be the same!!!")
    return theta

def subproblem2(p, q, k1, k2):
    eps = np.finfo(np.float64).eps
    norm = np.linalg.norm
    k12 = np.dot(k1, k2)
    pk = np.dot(p, k2)
    qk = np.dot(q, k1)
    # check if solution exists
    if (np.abs( 1 - k12**2) < eps):
        warnings.warn("No solution - k1 != k2")
        return []
    a = np.matmul([[k12, -1], [-1, k12]],[pk, qk]) / (k12**2 - 1)
    bb = (np.dot(p,p) - np.dot(a,a) - 2*a[0]*a[1]*k12)
    if (np.abs(bb) < eps): bb=0
    if (bb < 0):
        warnings.warn("No solution - no intersection found between cones")
        return []
    gamma = np.sqrt(bb) / norm(np.cross(k1,k2))
    if (np.abs(gamma) < eps):
        cm=np.array([k1, k2, np.cross(k1,k2)]).T
        c1 = np.dot(cm, np.hstack((a, gamma)))
        theta2 = subproblem1(k2, p, c1)
        theta1 = -subproblem1(k1, q, c1)
        return [(theta1, theta2)]
    cm=np.array([k1, k2, np.cross(k1,k2)]).T
    c1 = np.dot(cm, np.hstack((a, gamma)))
    c2 = np.dot(cm, np.hstack((a, -gamma)))
    theta1_1 = -subproblem1(q, c1, k1)
    theta1_2 = -subproblem1(q, c2, k1)
    theta2_1 =  subproblem1(p, c1, k2)
    theta2_2 =  subproblem1(p, c2, k2)
    return [(theta1_1, theta2_1), (theta1_2, theta2_2)]


def subproblem3(p, q, k, d):
    norm=np.linalg.norm
    pp = np.subtract(p,np.dot(np.dot(p, k),k))
    qp = np.subtract(q,np.dot(np.dot(q, k),k))
    dpsq = d**2 - ((np.dot(k, np.add(p,q)))**2)
    bb=-(np.dot(pp,pp) + np.dot(qp,qp) - dpsq)/(2*norm(pp)*norm(qp))
    if dpsq < 0 or np.abs(bb) > 1:
        warnings.warn("No solution - no rotation can achieve specified distance")
        return []
    theta = subproblem1(pp/norm(pp), qp/norm(qp), k)
    phi = np.arccos(bb)
    if np.abs(phi) > 0:
        return [theta + phi, theta - phi]
    else:
        return [theta]

def subproblem4(p, q, k, d):
    a = np.matmul(np.matmul(p,hat(k)),q)
    b = -np.matmul(p, np.matmul(hat(k),hat(k).dot(q)))
    c = np.subtract(d, (np.dot(p,q) -b))
    phi = np.arctan2(b, a)
    d = c / np.linalg.norm([a,b])
    if d > 1:
        return []
    psi = np.arcsin(d)
    return [-phi+psi, -phi-psi+np.pi]