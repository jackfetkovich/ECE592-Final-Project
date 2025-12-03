import mujoco
import numpy as np
from numba import int64, float64, boolean
from numba.experimental import jitclass
from numba import njit
from transform import *
from dynamics import *
import csv

allocation_matrix = -np.array([
    [-0.4472, -0.4472, 0, 0, -0.4472, -0.4472, 0, 0],
    [0.8944, -0.8944, 0, 0, 0.8944, -0.8944, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, -0.2828, -0.2828, 0, 0, 0.2828, 0.2828],
    [0, 0, 0.2828, -0.2828, 0, 0, -0.2828, 0.2828],
    [0.4472, -0.4472, 0, 0, -0.4472, 0.4472, 0, 0]
])

filename = "./debug_data/mppi_debug.csv"


def mppi_mujoco_parallel(x_init, v_init, traj, time, K, T, lam, dt, model_headless, data_headless):    
    # Preallocate candidate control sequences
    means = np.array([0,0,223,0,0,0])
    sigmas = np.array([1000,1000,1000,0.0,0.0,600.0])
    
    U = means + sigmas * np.random.randn(K, T, 6)
    alloc_inv = np.linalg.pinv(allocation_matrix)
    
    # Discretize trajectory
    targets = np.zeros((T, 6))
    for t in range(T):
        targets[t] = traj.sample_trajectory(time + t*dt)

    
    # Preallocate costs
    costs = np.zeros(K)
    # Parallel loop: candidate trajectories
    for k in range(K):


        # Reset headless MuJoCo for this trajectory
        mujoco.mj_resetData(model_headless, data_headless)
        data_headless.qpos[:7] = x_init
        data_headless.qvel[:6] = v_init
        mujoco.mj_forward(model_headless, data_headless)
        # print(U[k, :])
        
        for t in range(T):
            data_headless.ctrl[:8] = alloc_inv @ U[k,t]
            mujoco.mj_step(model_headless, data_headless)
            
            x_t = data_headless.qpos[:7].copy()
            u_t = U[k,t]
            this_cost = cost_function(x_t, u_t, targets[t])

            costs[k] += this_cost
        # with open(filename, 'a', newline='', encoding='utf-8') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([x_t[2], targets[t,2], abs(targets[t,2] - x_t[2]), this_cost])
        
        costs[k] += terminal_cost(data_headless.qpos[:6], targets[-1])
        # print(costs[k])
    
    # Compute weights
    weights = np.exp(-(costs - np.min(costs))/lam)
    sum_weights = np.sum(weights)
    if sum_weights < 1e-10:
        weights = np.ones_like(weights)/len(weights)
        print("NUMERICAL ISSUE")
    else:
        weights /= sum_weights

    
    
    # Weighted sum of control sequences
    u_star = np.sum(weights[:, None, None]*U, axis=0)
    print (u_star[0, 5])
    return u_star[0]


def mujoco_rollout(model, data, eta_init, v_init, U, dt):
    """
    Simulate a sequence of controls in headless MuJoCo.
    """

    T = U.shape[0]
    trajectory = np.zeros((T+1, 12))
    
    # Initialize
    data.qpos[:6] = eta_init
    data.qvel[:6] = v_init
    mujoco.mj_forward(model, data)
    
    trajectory[0, :6] = eta_init
    trajectory[0, 6:] = v_init
    
    for t in range(T):
        # Apply control
        data.ctrl[:6] = U[t]
        
        # Step MuJoCo
        mujoco.mj_step(model, data)
        
        # Save state
        trajectory[t+1, :6] = data.qpos[:6].copy()
        trajectory[t+1, 6:] = data.qvel[:6].copy()
        
    return trajectory

# Cost function

def cost_function(x_in, u, target):
    x = np.concatenate([x_in[:3],quat_to_euler_xyz(x_in[3:7])])
    Q = np.diag(np.array([10.0,10.0, 10.0, 0.0, 0.0, 8.0]))           # State costs
    R = np.diag(np.array([1e-13,1e-13, 1e-13, 1e-13, 1e-13, 1e-13]))  # Input costs

    x_des = np.array([target[0], target[1], target[2], target[3], target[4], target[5]])
    state_diff = x_des - x
    for i in range(3, 6, 1):
        state_diff[i] = wrap_angle(state_diff[i])
    
    state_cost = np.dot(state_diff.T, np.dot(Q,state_diff))

    cost = state_cost + np.dot(u.T, np.dot(R, u))
    return cost

# Terminal Cost Function

def terminal_cost(x_in, target):
    x = np.concatenate([x_in[:3], quat_to_euler_xyz(x_in[3:7])])
    Q = np.diag(np.array([12.0, 12.0, 12.0, 0.0, 0.0,10.0]))  # State costs
    x_des = np.array([target[0], target[1], target[2], target[3], target[4], target[5]])
    state_diff = x_des - x
    for i in range(3, 6, 1):
        state_diff[i] = wrap_angle(state_diff[i])
    terminal_cost = np.dot(state_diff.T, np.dot(Q,state_diff))
    return terminal_cost 

@njit
def gen_normal_control_seq(means, sigmas, K, T):
    return np.dstack((
        np.random.normal(loc=means[0], scale=sigmas[0], size=(K, T)), #Fx
        np.random.normal(loc=means[1], scale=sigmas[1], size=(K, T)), #Fy
        np.random.normal(loc=means[2], scale=sigmas[2], size=(K, T)), #Fz
        np.random.normal(loc=means[3], scale=sigmas[3], size=(K, T)), #phi_dot
        np.random.normal(loc=means[4], scale=sigmas[4], size=(K, T)), #theta_dot
        np.random.normal(loc=means[5], scale=sigmas[5], size=(K, T)), #psi_dot
    )) 
