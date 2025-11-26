import mujoco
import mujoco.viewer
import time
import numpy as np
import os
from sub import Sub
from telemetry import Telemetry
import matplotlib
import matplotlib.pyplot as plt
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(BASE_DIR, "project.xml")


model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
# data.qvel[3:6] = 5*np.random.randn(3)

def init_swix():
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

    telemetry = Telemetry(data, model)

    return Sub(eta, v, Mrb, Crb, Ma, Ca, D, g, iface, telemetry)

sub = init_swix()
predicted_state = sub.eta
t = []
predicted_states = []
true_states = []
plotting_started = True

ctrl = [-3000, -1000, 0, 400, 300, -1000]

start = time.perf_counter()
with mujoco.viewer.launch_passive(model, data) as viewer:
    dt = model.opt.timestep
    while viewer.is_running():
        now = time.perf_counter()
        # Apply controller input
        sub.control(ctrl)
        mujoco.mj_step(model, data)

        if now - start > 5 and not plotting_started:
            predicted_state = np.concatenate([sub.telemetry.pos(), sub.telemetry.rot()]).tolist()
            plotting_started = True
        
        if plotting_started:
            predicted_state = sub.forward_dynamics(predicted_state, sub.v, np.array(ctrl), dt)
            # Save predicted and true measurements
            predicted_states.append(predicted_state)
            true_state = np.concatenate([sub.telemetry.pos(), sub.telemetry.rot()]).tolist()
            true_states.append(true_state)
            t.append(now-start)

        viewer.sync()
        
        if now - start > 10:
            break
        
        time.sleep(dt)

    
predicted_states = np.array(predicted_states)
true_states = np.array(true_states)

plt.subplot(231)
plt.plot(t, predicted_states[:, 0], label="Predicted")
plt.plot(t, true_states[:, 0], label="Measured")
plt.legend()

plt.subplot(232)
plt.plot(t, predicted_states[:, 1])
plt.plot(t, true_states[:, 1])

plt.subplot(233)
plt.plot(t, predicted_states[:, 2])
plt.plot(t, true_states[:, 2])

plt.subplot(234)
plt.plot(t, predicted_states[:, 3])
plt.plot(t, true_states[:, 5])

plt.subplot(235)
plt.plot(t, predicted_states[:, 4])
plt.plot(t, true_states[:, 4])

plt.subplot(236)
plt.plot(t, predicted_states[:, 5])
plt.plot(t, true_states[:, 3])

plt.show()

