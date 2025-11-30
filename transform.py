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

def euler_xyz_to_quat(euler):
    """
    Convert Euler angles [roll, pitch, yaw] (x, y, z) to quaternion [w, x, y, z].
    Assumes the same convention as quat_to_euler_xyz: rotations about x, then y, then z.
    """
    roll  = euler[0]  # phi
    pitch = euler[1]  # theta
    yaw   = euler[2]  # psi

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy

    return np.array([w, x, y, z])

def euler_zyx_rates_to_body_omega(euler, euler_rates):
    """
    Convert ZYX Euler angle rates (roll = phi, pitch = theta, yaw = psi)
    into body-frame angular velocity [wx, wy, wz].

    euler = [phi, theta, psi]
    euler_rates = [phi_dot, theta_dot, psi_dot]
    """

    phi, theta, psi = euler
    p, q, r = euler_rates   # Euler rates: φ̇, θ̇, ψ̇

    # Build transformation for ZYX Euler angles
    # ω = T(φ,θ) * [φ̇, θ̇, ψ̇]
    wx = p - r * np.sin(theta)
    wy = q * np.cos(phi) + r * np.sin(phi) * np.cos(theta)
    wz = -q * np.sin(phi) + r * np.cos(phi) * np.cos(theta)

    return np.array([wx, wy, wz])

def euler_zyx_to_quat(euler):
    """
    Convert ZYX Euler angles [roll, pitch, yaw] (phi, theta, psi)
    into a quaternion [w, x, y, z].

    Rotation order: yaw (Z), pitch (Y), roll (X)
    """
    phi   = euler[0]   # roll
    theta = euler[1]   # pitch
    psi   = euler[2]   # yaw

    c1 = np.cos(psi   * 0.5)   # yaw
    s1 = np.sin(psi   * 0.5)
    c2 = np.cos(theta * 0.5)   # pitch
    s2 = np.sin(theta * 0.5)
    c3 = np.cos(phi   * 0.5)   # roll
    s3 = np.sin(phi   * 0.5)

    # ZYX quaternion construction
    w = c1*c2*c3 + s1*s2*s3
    x = c1*c2*s3 - s1*s2*c3
    y = c1*s2*c3 + s1*c2*s3
    z = s1*c2*c3 - c1*s2*s3

    return np.array([w, x, y, z])