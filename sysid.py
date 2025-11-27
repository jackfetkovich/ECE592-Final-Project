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
body_id = model.body(name="SWIX").id
m   = model.body_mass[body_id]
I_mj   = model.body_inertia[body_id]     # [Ixx, Iyy, Izz]

Ix, Iy, Iz = I_mj[0], I_mj[1], I_mj[2]
print(I_mj)

grav = 9.81
vol = 0.128
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
    [0.0,0.0,3.0,0.0,0.0,0.0, 1.0],
    [0.0,0.0,3.0,0.0,0.0,0.0, 2.0],
    [0.0,0.0,5.0,0.0,0.0,0.0, 3.0]
])

traj = Trajectory(waypoints)

def warmup():
    print("Warming up...")
    mppi(np.concat([eta, np.zeros(6)]), traj, 0.15, 1000, 12, 1, 0.1, m, Ix, Iy, Iz, W, B, params)

# warmup()


predicted_state = np.concatenate((eta, v)).astype(np.float64)

t = []
goal_states = []
true_states = []
plotting_started = True
sim_time = 0

# ctrl = np.zeros(6)
count = 0
start = time.perf_counter()
with mujoco.viewer.launch_passive(model, data) as viewer:
    dt = model.opt.timestep
    while viewer.is_running():
       
        ctrl = np.random.uniform([0, 0, 0, 0, 2000, 0])
        sub.control(ctrl)
        mujoco.mj_step(model, data)
        print(sub.telemetry.vel())
        now = time.perf_counter()
        # print(now-start)

        #  Apply static input

        # if count % 100 == 0:
        #     ctrl = mppi(sub.telemetry.x(), traj, sim_time, 2500, 20, 1, 0.01, m, Ix, Iy, Iz, W, B, params)


        goal_states.append(predicted_state.copy())

        true_state = np.concatenate([sub.telemetry.pos(), sub.telemetry.rot()]).tolist()

        true_states.append(true_state)
        t.append(sim_time)

        viewer.sync()
        
        if now - start > 15:
            break
        count +=1 
        sim_time += dt
        time.sleep(dt)

    
predicted_states = np.array(goal_states)
true_states = np.array(true_states)
print(sim_time)

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