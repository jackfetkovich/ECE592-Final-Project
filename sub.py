import numpy as np
from numpy import cos as c
from numpy import sin as s
from numpy import tan as t
from numba import int64, float64, boolean
from numba.experimental import jitclass
from numba import njit
from transform import *
from telemetry import Telemetry



spec = []
@jitclass(spec)
class SubParams:
    def __init__(self, Mrb, Crb, Ma, Ca, D, g):
        self.Mrb = Mrb
        self.Crb = Crb
        self.Ma = Ma
        self.Ca = Ca
        self.D = D
        self.g = g


class Sub:
    def __init__(self, eta0, v0, ctrl_iface, telemetry, params):
        self.eta = eta0 # set initial state
        self.eta_dot = self.J(eta0) @ v0
        self.x = np.array([self.eta, self.eta_dot])
        self.v = v0
        self.ctrl_iface = ctrl_iface
        self.telemetry = telemetry
        self.params = params
        self.allocation_matrix = -np.array([
            [-0.4472, -0.4472, 0, 0, -0.4472, -0.4472, 0, 0],
            [0.8944, -0.8944, 0, 0, 0.8944, -0.8944, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, -0.2828, -0.2828, 0, 0, 0.2828, 0.2828],
            [0, 0, 0.2828, -0.2828, 0, 0, -0.2828, 0.2828],
            [0.4472, -0.4472, 0, 0, -0.4472, 0.4472, 0, 0]
        ])
        
    def control(self, tau):
        self.v = self.telemetry.vel()
        thrusts = np.linalg.pinv(self.allocation_matrix) @ tau
        self.ctrl_iface(thrusts)

    def print_telemetry(self):
        pos = self.telemetry.pos()
        rot = self.telemetry.rot()
        vel = self.telemetry.vel()
        print("******************************")
        print(f"Pos: ({pos[0]}, {pos[1]}, {pos[2]})")
        print(f"Rot: ({rot[0]}, {rot[1]}, {rot[2]})")
        print(f"Vel: ({vel[0]}, {vel[1]}, {vel[2]})")