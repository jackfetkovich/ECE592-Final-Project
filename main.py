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

eta = np.array([0, 0, 5, 0, 0, 0])
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
    [3.0,0.0,8.0,0.0,0.0,0.0, 4.0],
    [7.0,4.0,8.0,0.0,0.0,0.0, 8.0],
    [4.0,-2.0,5.0,0.0,0.0,0.0, 12.0]
])

traj = Trajectory(waypoints)


predicted_state = sub.eta
t = []
goal_states = []
true_states = []
plotting_started = True

ctrl = np.zeros(6)
count = 0
start = time.perf_counter()
with mujoco.viewer.launch_passive(model, data) as viewer:
    dt = model.opt.timestep
    while viewer.is_running():
       
        now = time.perf_counter()
        print(now-start)
        if count % 1000 == 0:
            ctrl = mppi(sub.telemetry.x(), traj, now-start, 2000, 25, 1, 0.01, m, Ix, Iy, Iz, W, B, params)
        # Apply controller input
        sub.control(ctrl)
        mujoco.mj_step(model, data)

        goal_state = traj.sample_trajectory(now-start)
        # Save predicted and true measurements
        goal_states.append(goal_state)
        true_state = np.concatenate([sub.telemetry.pos(), sub.telemetry.rot()]).tolist()
        true_states.append(true_state)
        t.append(now-start)

        viewer.sync()
        
        if now - start > 30:
            break
        count +=1 
        time.sleep(dt)

    
predicted_states = np.array(goal_states)
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

