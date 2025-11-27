import mujoco
import mujoco.viewer
import time
import numpy as np
import os
from sub import Sub, SubParams
from telemetry import Telemetry
import matplotlib.pyplot as plt
import time
from mppi import *
from mppi import mppi_mujoco_parallel
from trajectory import Trajectory
from dynamics import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(BASE_DIR, "project.xml")


model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

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

eta = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
v = np.zeros(6)
Mrb = np.diag([m, m, m, Ix, Iy, Iz])

def iface(thrusts):
    for i, thrust in enumerate(thrusts):
        data.ctrl[i] = thrust

telemetry = Telemetry(data, model)
params = SubParams(Mrb, Ma)

sub = Sub(eta, v, iface, telemetry, params)

waypoints = np.array([
    [0.0,0.0,5.0,0.0,0.0,0.0, 0.0],
    [0.0,0.0,8.0,0.0,0.0,0.0, 4.0],
    [0.0,0.0,8.0,0.0,0.0,0.0, 8.0],
    [0.0,0.0,5.0,0.0,0.0,0.0, 12.0]
])

traj = Trajectory(waypoints)

# Headless MuJoCo instance for MPPI
model_headless = mujoco.MjModel.from_xml_path(xml_path)
data_headless = mujoco.MjData(model_headless)

def warmup():
    print("Warming up...")
    # mppi(np.concat([eta, np.zeros(6)]), traj, 0.15, 1000, 12, 1, 0.1, m, Ix, Iy, Iz, W, B, params)
    mppi_mujoco_parallel(
        np.concatenate([eta, np.zeros(6)]),  # initial state [eta | v]
        traj,
        0.15,                                # starting time
        K=50,                                # small number for warmup
        T=12,
        lam=1.0,
        dt=model.opt.timestep,
        model_headless=model_headless,
        data_headless=data_headless
    )

warmup()


actual_state = np.concatenate((eta, v)).astype(np.float64)

t = []
actual_states = []
desired_states = []
plotting_started = True
sim_time = 0

count = 0
start = time.perf_counter()
ctrl = np.array([300.0,0.0,0.0,0.0,0.0,0.0])
with mujoco.viewer.launch_passive(model, data) as viewer:
    dt = model.opt.timestep
    while viewer.is_running():
       
        now = time.perf_counter()
        print(now-start)

        #  Apply static input

        if count % 100 == 0:
        #     ctrl = mppi(sub.telemetry.x(), traj, sim_time, 2500, 20, 1, 0.01, m, Ix, Iy, Iz, W, B, params)
            ctrl = mppi_mujoco_parallel(
                np.concatenate([sub.eta, sub.v]),
                traj,
                sim_time,
                K=200,            # candidate trajectories
                T=20,              # horizon
                lam=1.0,
                dt=model.opt.timestep,
                model_headless=model_headless,
                data_headless=data_headless
            )

        # Apply controller input
        sub.control(ctrl)
        mujoco.mj_step(model, data)

        # goal_state = traj.sample_trajectory(sim_time)

        actual_eta = actual_state[:6]
        actual_v = actual_state[6:]

        actual_state = np.concatenate([sub.telemetry.pos(), sub.telemetry.rot()]).astype(np.float64)

        # goal_states.append(goal_state)
        actual_states.append(actual_state.copy())

        # desired state from defined trajectory
        desired_state = traj.sample_trajectory(sim_time)
        desired_states.append(desired_state)

        t.append(sim_time)

        viewer.sync()
        
        # simulation length
        if now - start > 60:
            break
        count +=1 
        sim_time += dt
        time.sleep(dt)

    
# predicted_states = np.array(goal_states)
actual_states = np.array(actual_states)

# true_states = np.array(true_states)
desired_states = np.array(desired_states)
print(sim_time)

plt.subplot(231)
# plt.plot(t, predicted_states[:, 0], label="Predicted")
plt.plot(t, actual_states[:, 0], label="Actual")

# plt.plot(t, true_states[:, 0], label="Measured")
plt.plot(t, desired_states[:, 0], label="Desired")

plt.legend()

plt.subplot(232)
# plt.plot(t, predicted_states[:, 1])
plt.plot(t, actual_states[:, 1])
plt.plot(t, desired_states[:, 1])

plt.subplot(233)
# plt.plot(t, predicted_states[:, 2])
plt.plot(t, actual_states[:, 2])
plt.plot(t, desired_states[:, 2])

plt.subplot(234)
# plt.plot(t, predicted_states[:, 3])
plt.plot(t, actual_states[:, 3])
plt.plot(t, desired_states[:, 3])

plt.subplot(235)
# plt.plot(t, predicted_states[:, 4])
plt.plot(t, actual_states[:, 4])
plt.plot(t, desired_states[:, 4])

plt.subplot(236)
# plt.plot(t, predicted_states[:, 5])
plt.plot(t, actual_states[:, 5])
plt.plot(t, desired_states[:, 5])

plt.show()

