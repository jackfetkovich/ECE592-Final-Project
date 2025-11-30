import numpy as np
from numpy import cos as c
from numpy import sin as s
from numpy import tan as t
from numba import njit

@njit
def rotZ(psi):
    return np.array([
        [c(psi), -s(psi), 0],
        [s(psi), c(psi), 0],
        [0, 0, 1]
    ])

@njit
def rotX(phi):
    return np.array([
        [1, 0, 0],
        [0, c(phi), -s(phi)],
        [0, s(phi), c(phi)],  
    ])

@njit
def rotY(theta):
    return np.array([
        [c(theta), 0, s(theta)],
        [0, 1, 0],
        [-s(theta), 0, c(theta)] 
    ])

@njit
def trfm(x, y, z, phi, theta, psi):
    rot = rotZ(psi)@rotY(theta)@rotX(phi)
    trans = np.array([[x], [y], [z]])

    return np.block([
        [rot, trans],
        [0, 0, 0, 1]
    ])

@njit
def quat_to_euler_xyz(q):
    # q = [w, x, y, z]
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    # ----- yaw (Z rotation) -----
    t0 = 2.0 * (w*z + x*y)
    t1 = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.arctan2(t0, t1)

    # ----- pitch (Y rotation) -----
    sinp = 2.0 * (w*y - z*x)

    # manual clamp because np.clip is not allowed in nopython
    if sinp > 1.0:
        sinp = 1.0
    elif sinp < -1.0:
        sinp = -1.0

    pitch = np.arcsin(sinp)

    # ----- roll (X rotation) -----
    t2 = 2.0 * (w*x + y*z)
    t3 = 1.0 - 2.0 * (x*x + y*y)
    roll = np.arctan2(t2, t3)

    # ---- output ----
    out = np.zeros(3)
    out[2] = yaw
    out[1] = pitch
    out[0] = roll
    return out

@njit
def wrap_angle(a):
    # works on scalars or numpy arrays
    return (a + np.pi) % (2*np.pi) - np.pi