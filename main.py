import mujoco
import mujoco.viewer
import time
import numpy as np
import os
from sub import Sub
from telemetry import Telemetry

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(BASE_DIR, "project.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
# data.qvel[3:6] = 5*np.random.randn(3)

def init_swix():
    # Initialize submarine
    m = 100
    Ix = 10
    Iy = 10
    Iz = 10

    grav = 9.81
    vol = 0.178
    rho = 1000

    W = m*grav
    B = rho * grav * vol

    eta = np.array([0, 0, 5, 0, 0, 0])
    v = np.zeros(6)
    Mrb = np.diag([m, m, m, Ix, Iy, Iz])
    Crb = lambda v: np.array([
        [0, 0, 0, 0, v[2], 0],
        [0, 0, 0, -m * v[2], 0, 0],
        [0, 0, 0, m*v[1], -m*v[1], 0],
        [0, m*v[2], -m*v[1], 0, Iz * v[5], -Iy*v[4]],
        [-m*v[2], 0, -m*v[0], -Iz * v[5], 0, Ix*v[3]],
        [-m*v[1], -m*v[0], 0, Iy*v[4], -Ix*v[3], 0]
    ])
    Ma = np.zeros((6,6))
    Ca = lambda v: np.zeros((6,6))
    D = lambda v: np.zeros((6,6))
    g = lambda eta: np.array([
        (W-B)*np.sin(eta[4]),
        -(W-B)*np.cos(eta[4])*np.sin(eta[3]),
        -(W-B)*np.cos(eta[4])*np.cos(eta[3]),
        0,
        0, 
        0
    ])

    def iface(thrusts):
        for i, thrust in enumerate(thrusts):
            data.ctrl[i] = thrust

    telemetry = Telemetry(data)

    return Sub(eta, v, Mrb, Crb, Ma, Ca, D, g, iface, telemetry)

sub = init_swix()

x_new = sub.forward_dynamics(sub.eta, sub.v, np.array([100, 0, 0, 0, 0, 0]), 4)
print(x_new)

with mujoco.viewer.launch_passive(model, data) as viewer:
    dt = model.opt.timestep
    while viewer.is_running():
        # data.ctrl[0] = 500
        # data.ctrl[1] = 500
        # data.ctrl[2] = 0
        # data.ctrl[3] = 0
        # data.ctrl[4] = 500
        # data.ctrl[5] = 500
        # data.ctrl[6] = 0
        # data.ctrl[7] = 0
        sub.control([1,0,0,0,0,0])
        sub.print_telemetry()
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)