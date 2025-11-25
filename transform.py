import numpy as np
from numpy import cos as c
from numpy import sin as s
from numpy import tan as t

def rotZ(psi):
    return np.array([
        [c(psi), -s(psi), 0],
        [s(psi), c(psi), 0],
        [0, 0, 1]
    ])

def rotX(phi):
    return np.array([
        [1, 0, 0],
        [0, c(phi), -s(phi)],
        [0, s(phi), c(phi)],  
    ])

def rotY(theta):
    return np.array([
        [c(theta), 0, s(theta)],
        [0, 1, 0],
        [-s(theta), 0, c(theta)] 
    ])

def trfm(x, y, z, phi, theta, psi):
    rot = rotZ(psi)@rotY(theta)@rotX(phi)
    trans = np.array([[x], [y], [z]])

    return np.block([
        [rot, trans],
        [0, 0, 0, 1]
    ])

def quat_to_euler_zyx(q):
    """
    Convert a unit quaternion q = [w, x, y, z] to Euler angles
    (yaw, pitch, roll) in radians using ZYX order.
    """

    w, x, y, z = q

    # yaw (Z axis rotation)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    # pitch (Y axis rotation)
    sinp = 2*(w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)  # numeric safety
    pitch = np.arcsin(sinp)

    # roll (X axis rotation)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))

    return np.array([yaw, pitch, roll])