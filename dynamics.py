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

@njit
def inertia_box_radii(m, Ix, Iy, Iz):
    rx = np.sqrt(3.0/(2.0*m) * (Iy + Iz - Ix))
    ry = np.sqrt(3.0/(2.0*m) * (Iz + Ix - Iy))
    rz = np.sqrt(3.0/(2.0*m) * (Ix + Iy - Iz))
    return rx, ry, rz

@njit
def fluid_forces_inertia_model(v_body, omega_body,
                               m, Ix, Iy, Iz,
                               rho, beta):
    """
    v_body, omega_body: 3-vectors in BODY frame, matching MuJoCo.
    rho: fluid density (MuJoCo 'density')
    beta: viscosity (MuJoCo 'viscosity')
    """

    rx, ry, rz = inertia_box_radii(m, Ix, Iy, Iz)
    # rx = 2*rx
    # ry = 2*ry
    # rz = 2*rz
    req = (rx + ry + rz) / 3.0

    r = np.array((rx, ry, rz))

    # subtract wind in body frame
    v = np.empty(3)
    v[0] = v_body[0]
    v[1] = v_body[1]
    v[2] = v_body[2]

    w = omega_body

    f = np.zeros(3)
    g = np.zeros(3)

    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3

        vi = v[i]
        wi = w[i]

        # Quadratic drag
        f[i] += -2.0 * rho * r[j] * r[k] * np.abs(vi) * vi
        g[i] += -0.5 * rho * r[i] * (r[j]**4 + r[k]**4) * np.abs(wi) * wi

        # Viscous drag
        f[i] += -6.0 * beta * np.pi * req * vi
        g[i] += -8.0 * beta * np.pi * (req**3) * wi

    tau_fluid = np.zeros(6)
    tau_fluid[0] = f[0]
    tau_fluid[1] = f[1]
    tau_fluid[2] = f[2]
    tau_fluid[3] = g[0]
    tau_fluid[4] = g[1]
    tau_fluid[5] = g[2]

    return tau_fluid

Ma = np.zeros((6,6))

@njit("float64[:,:](float64[:])")
def D(v): 
    return np.zeros((6,6))

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


@njit("float64[:](float64[:], float64, float64)")
def g(eta, W, B): 
    # With z-up world:
    Fz_world = B-W

    phi   = eta[3]
    theta = eta[4]
    psi   = eta[5]

    # Rotate world force into body frame (inverse of r_b_to_n)
    R = r_b_to_n(phi, theta, psi)   # body → world
    F_body = R.T @ np.array([0.0, 0.0, Fz_world])

    return np.array([F_body[0], F_body[1], F_body[2], 0.0, 0.0, 0.0])

@njit
def forward_dynamics(eta, v, tau, dt, m, Ix, Iy, Iz, W, B, params):
    v_lin = v[:3]
    v_ang = v[3:]
    # MuJoCo-style fluid generalized forces in BODY frame
    tau_fluid = fluid_forces_inertia_model(v_lin, v_ang,
                                           m, Ix, Iy, Iz,
                                           1000, 0.000001)
    v_dot = np.linalg.inv(params.Mrb) @ (tau - g(eta, W, B) + Crb(v, m, Ix, Iy, Iz)@v + tau_fluid)
    v = v + v_dot * dt
    eta_dot = J(eta) @ v
    eta_next = eta[0:6] + eta_dot * dt

    eta_next[3:6] = wrap_angle(eta_next[3:6])

    return concat1d(eta_next, v)

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