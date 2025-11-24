import numpy as np
from numpy import cos as c
from numpy import sin as s
from numpy import tan as t
from numba import int64, float64, boolean
from numba.experimental import jitclass

spec = []


"""
 All parameters needed to define sub
    - X (x, y, z, phi, theta, psi, u, v, w, p, q, r) initial
    - Mass matrix
    - Thruster allocation matrix
    - 

"""

class Sub:
    def __init__(self, eta, v, Mrb, Crb, Ma, Ca, D, g):
        self.eta = eta # set initial state
        self.v = v
        self.Mrb = Mrb
        self.Crb = Crb
        self.Ma = Ma
        self.Ca = Ca
        self.D = D
        self.g = g
        self.eta = eta

    def forward_dynamics(self, eta, v, tau, dt):
        """_summary_

        Args:
            x ((6, ) numpy array): Starting state of robot
            tau ((6, ) numpy array): _description_
        """
        
        v_dot = np.linalg.inv(self.Mrb + self.Ma) @ (tau - self.g(self.eta) - (self.Crb(v)@v + self.Ca(v)@v + self.D(v)@v))
        print(f"vdot: {v_dot}")
        v = v + v_dot * dt
        print(f"v: {v}")
        eta_dot = self.J(v) @ eta
        print(f"eta_dot: {eta_dot}")
        return eta + eta_dot * dt

    def r_b_to_n(self, phi, theta, psi):
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
        rbn = self.r_b_to_n(phi, theta, psi)
        T = self.omega_to_world(phi, theta, psi)
        zeros = np.zeros((3,3))
        
        return np.block([
            [rbn, zeros],
            [zeros, T]
        ])








