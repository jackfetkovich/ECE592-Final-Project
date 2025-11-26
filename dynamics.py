import numpy as np
from numpy import cos as c
from numpy import sin as s
from numpy import tan as t
from numba import int64, float64, boolean
from numba import njit
from transform import *

# Initialize submarine
m = 22.0
Ix = 1/12 * m * (0.4**2 + 0.4**2) 
Iy = 1/12 * m * (0.4**2 + 0.8**2)
Iz = 1/12 * m * (0.8**2 + 0.4**2)

grav = 9.81
vol = 0.178
rho = 1000

W = m*grav
B = rho * grav * vol

eta = np.array([0, 0, 5, 0, 0, 0])
v = np.zeros(6)
Mrb = np.diag([m, m, m, Ix, Iy, Iz])

@njit("float64(float64[:,:])")
def Crb(v): 
    return np.array([
        [0, 0, 0, 0, v[2], 0],
        [0, 0, 0, -m * v[2], 0, 0],
        [0, 0, 0, m*v[1], -m*v[1], 0],
        [0, m*v[2], -m*v[1], 0, Iz * v[5], -Iy*v[4]],
        [-m*v[2], 0, -m*v[0], -Iz * v[5], 0, Ix*v[3]],
        [-m*v[1], -m*v[0], 0, Iy*v[4], -Ix*v[3], 0]
    ])

Ma = np.zeros((6,6))

@njit("float64(float64[:,:])")
def Ca(v):
    return np.zeros((6,6))

@njit("float64(float64[:,:])")
def D(v): 
    return np.zeros((6,6))

@njit("float64[:](float64[:])")
def g(eta): 
    return np.array([
        (W-B)*np.sin(eta[4]),
        -(W-B)*np.cos(eta[4])*np.sin(eta[3]),
        -(W-B)*np.cos(eta[4])*np.cos(eta[3]),
        0,
        0, 
        0
    ])

@njit
def forward_dynamics(eta, v, tau, dt, Mrb, Ma, g, Crb, Ca, D):
    v_dot = np.linalg.inv(Mrb + Ma) @ (tau - g(eta) - (Crb(v)@v + Ca(v)@v + D(v)@v))
    v = v + v_dot * dt
    eta_dot = J(eta) @ v
    eta_next = eta[0:6] + eta_dot * dt

    eta_next[3:6] = wrap_angle(eta_next[3:6])

    return np.concatenate([eta_next, v])

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
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]
    rbn = r_b_to_n(phi, theta, psi) # Convert to z up, rotate to accomodate y being the forward axis
    T = omega_to_world(phi, theta, psi)
    zeros = np.zeros((3,3))
    
    return np.block([
        [rbn, zeros],
        [zeros, T]
    ])
