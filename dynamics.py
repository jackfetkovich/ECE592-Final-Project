import numpy as np
from numpy import cos as c
from numpy import sin as s
from numpy import tan as t
from numba import int64, float64, boolean
from numba import njit
from transform import *


@njit("float64[:,:](float64[:], float64, float64, float64, float64)")
def Crb(v, m, Ix, Iy, Iz): 
    return np.array([
        [0, 0, 0, 0, v[2], 0],
        [0, 0, 0, -m * v[2], 0, 0],
        [0, 0, 0, m*v[1], -m*v[1], 0],
        [0, m*v[2], -m*v[1], 0, Iz * v[5], -Iy*v[4]],
        [-m*v[2], 0, -m*v[0], -Iz * v[5], 0, Ix*v[3]],
        [-m*v[1], -m*v[0], 0, Iy*v[4], -Ix*v[3], 0]
    ])

Ma = np.zeros((6,6))

@njit("float64[:,:](float64[:])")
def Ca(v):
    return np.zeros((6,6))

@njit("float64[:,:](float64[:])")
def D(v): 
    return np.zeros((6,6))

@njit("float64[:](float64[:], float64, float64)")
def g(eta, W, B): 
    return np.array([
        (W-B)*np.sin(eta[4]),
        -(W-B)*np.cos(eta[4])*np.sin(eta[3]),
        -(W-B)*np.cos(eta[4])*np.cos(eta[3]),
        0,
        0, 
        0
    ])

@njit
def forward_dynamics(eta, v, tau, dt, m, Ix, Iy, Iz, W, B, params):
    v_dot = np.linalg.inv(params.Mrb + params.Ma) @ (tau - g(eta, W, B) - (Crb(v, m, Ix, Iy, Iz)@v + Ca(v)@v + D(v)@v))
    v = v + v_dot * dt
    eta_dot = J(eta) @ v
    eta_next = eta[0:6] + eta_dot * dt

    eta_next[3:6] = wrap_angle(eta_next[3:6])

    return concat1d(eta_next, v)

@njit
def r_b_to_n(phi, theta, psi):
    """ Transformation matrix from robot body frame to world frame

    Args:
        phi (float): Rotation about x
        theta (float): Rotation about y
        psi (float): Rotation about z

    Returns:
        3x3 Rotation Matrix
    """
    return np.array([
        [c(psi)*c(theta), -s(psi)*c(phi) + c(psi)*s(theta)*s(phi), s(psi)*s(phi)+c(psi)*c(phi)*s(theta) ],
        [s(psi)*c(theta), c(psi)*c(phi) + s(phi)*s(theta)*s(psi), -c(psi)*s(phi)+s(theta)*s(psi)*c(phi)],
        [-s(theta), c(theta)*s(phi), c(theta)*c(phi)]
    ])

@njit
def omega_to_world(phi, theta, psi):
    return np.array([
        [1, s(phi)*t(theta), c(phi)*t(theta)],
        [0, c(phi), -s(phi)],
        [0, s(phi)/c(theta), c(phi)/c(theta)]
    ])

@njit
def J(eta):
    phi   = eta[3]
    theta = eta[4]
    psi   = eta[5]

    rbn = r_b_to_n(phi, theta, psi)        # 3×3
    T   = omega_to_world(phi, theta, psi)  # 3×3

    Jmat = np.zeros((6, 6))

    # Top-left block = rbn
    for i in range(3):
        for j in range(3):
            Jmat[i, j] = rbn[i, j]

    # Bottom-right block = T
    for i in range(3):
        for j in range(3):
            Jmat[i+3, j+3] = T[i, j]

    return Jmat

@njit
def concat1d(a, b):
    out = np.zeros(a.size + b.size)
    out[:a.size] = a
    out[a.size:] = b
    return out