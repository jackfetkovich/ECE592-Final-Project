import numpy as np
from numpy import cos as c
from numpy import sin as s
from numpy import tan as t
from numba import int64, float64, boolean
from numba.experimental import jitclass
from transform import *
from telemetry import Telemetry

spec = []


"""
 All parameters needed to define sub
    - X (x, y, z, phi, theta, psi, u, v, w, p, q, r) initial
    - Mass matrix
    - Thruster allocation matrix
    - 

"""

class Sub:
    def __init__(self, eta0, v0, Mrb, Crb, Ma, Ca, D, g, ctrl_iface, telemetry):
        self.eta = eta0 # set initial state
        self.eta_dot = self.J(eta0) @ v0
        self.x = np.array([self.eta, self.eta_dot])
        self.v = v0
        self.Mrb = Mrb
        self.Crb = Crb
        self.Ma = Ma
        self.Ca = Ca
        self.D = D
        self.g = g
        self.ctrl_iface = ctrl_iface
        self.telemetry = telemetry
        self.allocation_matrix = np.array([
            [100,100,100,100,100,100],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1]
        ])
        

    def forward_dynamics(self, eta, v, tau, dt):
        """_summary_

        Args:
            x ((6, ) numpy array): Starting state of robot
            tau ((6, ) numpy array): _description_
        """
        
        v_dot = np.linalg.inv(self.Mrb + self.Ma) @ (tau - self.g(eta) - (self.Crb(v)@v + self.Ca(v)@v + self.D(v)@v))
        print(f"vdot: {v_dot}")
        v = v + v_dot * dt
        print(f"v: {v}")
        eta_dot = self.J(eta) @ v
        print(f"eta_dot: {eta_dot}")
        return eta + eta_dot * dt

    def r_b_to_n(self, phi, theta, psi):
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
    
    def omega_to_world(self, phi, theta, psi):
        return np.array([
            [1, s(phi)*t(theta), c(phi)*t(theta)],
            [0, c(phi), -s(phi)],
            [0, s(phi)/c(theta), c(phi)/c(theta)]
        ])
    
    def J(self, eta):
        phi = eta[3]
        theta = eta[4]
        psi = eta[5]
        rbn = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]) @ self.r_b_to_n(phi, theta, psi) # Convert to z up, rotate to accomodate y being the forward axis
        T = self.omega_to_world(phi, theta, psi)
        zeros = np.zeros((3,3))
        
        return np.block([
            [rbn, zeros],
            [zeros, T]
        ])
    
    def control(self, tau):
        thrusts = self.allocation_matrix @ tau
        self.ctrl_iface(thrusts)









