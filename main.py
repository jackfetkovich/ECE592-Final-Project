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
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(BASE_DIR, "scene.xml")

# Init Mujoco
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

# Trajectory Waypoints (x, y, z, roll, pitch, yaw, time)
waypoints = np.array([
    [0.0,0.0,5.0,0.0,0.0, 0, 0.0],                      # start at origin
    [3.0,1.5,3.5,0.0,0.0,0.0, 2.0],                     # through gate
    [6.0,1.5,2.5,0.0,0.0,-np.pi/2, 4.5],                # align with buoy
    [7.0, 1.0, 2.5, 0.0, 0.0, -3 * np.pi /4, 5.5],      # circumnavigate buoy clock-wise start
    [7.5, 0.7, 2.5, 0.0, 0.0, -0.99* np.pi, 6.5],
    [6.5, -1.0, 2.5, 0.0, 0.0, 3*np.pi/4, 7.5],
    [6.0, -1.5, 2.5, 0.0, 0.0, np.pi/2, 8.5],           # circumnavigate buoy clock-wise end
])

traj = Trajectory(waypoints)

# Headless MuJoCo instance for MPPI
model_headless = mujoco.MjModel.from_xml_path(xml_path)
data_headless = mujoco.MjData(model_headless)

def warmup():
    print("Warming up...")
    mppi_mujoco_parallel(
        np.zeros(7),                # initial state [eta | v]
        np.zeros(6),
        traj,
        0.15,                       # starting time
        K=2,                        # small number for warmup
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

filename = "./debug_data/xyz.csv"
with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'X Desired', 'X Actual','Y Desired', 'Y Actual', 'Z Desired', 'Z Actual',
                         'Roll Desired', 'Roll Actual','Pitch Desired', 'Pitch Actual', 'Yaw Desired', 'Yaw Actual'])

# Main simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    dt = model.opt.timestep
    while viewer.is_running():
       
        now = time.perf_counter()

        #  Apply static input

        # Run controller @ 10Hz
        if count % 100 == 0:
            ctrl = mppi_mujoco_parallel(
                data.qpos[:7],
                data.qvel[:6],
                traj,
                sim_time,
                K=800,            # candidate trajectories
                T=50,             # horizon
                lam=0.01,
                dt=model_headless.opt.timestep,
                model_headless=model_headless,
                data_headless=data_headless
            )
            actual_state = np.concatenate([sub.telemetry.pos(), sub.telemetry.rot()]).astype(np.float64)
            actual_states.append(actual_state.copy())

            # Desired state from defined trajectory
            desired_state = traj.sample_trajectory(sim_time)
            desired_states.append(desired_state)

            t.append(sim_time)
            with open(filename, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([sim_time, desired_state[0], actual_state[0], desired_state[1], actual_state[1], desired_state[2], actual_state[2],
                         desired_state[3], actual_state[3], desired_state[4], actual_state[4], desired_state[5], actual_state[5]])
        
        
        # Apply controller input
        sub.control(ctrl)
        mujoco.mj_step(model, data)


        viewer.sync()
        
        # Simulation length
        if sim_time >= 8.5:
            break
        count +=1 
        sim_time += dt
        time.sleep(dt)

    
## ===== POST SIMULATION PLOTTING ===== ##

actual_states = np.array(actual_states)
desired_states = np.array(desired_states)

# Plot X
plt.subplot(231)
plt.plot(t, actual_states[:, 0], label="Actual")
plt.plot(t, desired_states[:, 0], label="Desired")
plt.title("X")
plt.xlabel("t(s)")
plt.ylabel("x(m)")
plt.legend()


# Plot Y
plt.subplot(232)
plt.plot(t, actual_states[:, 1])
plt.plot(t, desired_states[:, 1])
plt.title("Y")
plt.xlabel("t(s)")
plt.ylabel("y(m)")
plt.legend()


# Plot Z
plt.subplot(233)
plt.plot(t, actual_states[:, 2])
plt.plot(t, desired_states[:, 2])
plt.title("Z")
plt.xlabel("t(s)")
plt.ylabel("z(m)")
plt.legend()


# Plot Roll
plt.subplot(234)
plt.plot(t, actual_states[:, 3])
plt.plot(t, desired_states[:, 3])
plt.title("Roll")
plt.xlabel("t(s)")
plt.ylabel("Roll(rad)")
plt.legend()

# Plot Pitch
plt.subplot(235)
plt.plot(t, actual_states[:, 4])
plt.plot(t, desired_states[:, 4])
plt.title("Pitch")
plt.xlabel("t(s)")
plt.ylabel("Pitch(rad)")
plt.legend()

# Plot Yaw
plt.subplot(236)
plt.plot(t, actual_states[:, 5])
plt.plot(t, desired_states[:, 5])
plt.title("Yaw")
plt.xlabel("t(s)")
plt.ylabel("Yaw(rad)")
plt.legend()

plt.show()

