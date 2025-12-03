import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('./debug_data/xyz.csv')

t = data["Time"].to_numpy()
x_des = data["X Desired"].to_numpy()
x = data["X Actual"].to_numpy()
y_des = data["Y Desired"].to_numpy()
y = data["Y Actual"].to_numpy()
z_des = data["Z Desired"].to_numpy()
z = data["Z Actual"].to_numpy()
r_des = data["Roll Desired"].to_numpy()
r = data["Roll Actual"].to_numpy()
p_des = data["Pitch Desired"].to_numpy()
p = data["Pitch Actual"].to_numpy()
yaw_des = data["Yaw Desired"].to_numpy()
yaw = data["Yaw Actual"].to_numpy()

actual_states = np.column_stack([x, y, z, r, p, yaw])
desired_states = np.column_stack([x_des, y_des, z_des, r_des, p_des, yaw_des])

fig = plt.figure(figsize=(12, 10))
fig.suptitle(r"Controller Tracking Performance for X = (x, y, z, $\phi$, $\theta$, $\psi$)", y=0.98)

# Axes list for convenience
axes = [
    plt.subplot(321),
    plt.subplot(323),
    plt.subplot(325),
    plt.subplot(322),
    plt.subplot(324),
    plt.subplot(326)
]

names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
ylabels = ["x (m)", "y (m)", "z (m)", "Roll (rad)", "Pitch (rad)", "Yaw (rad)"]

for i, ax in enumerate(axes):
    ax.plot(t, actual_states[:, i], label=names[i])
    ax.plot(t, desired_states[:, i], label=f"{names[i]} Traj")
    ax.set_title(names[i])
    ax.set_xlabel("t (s)")
    ax.set_ylabel(ylabels[i])
    if i >= 3:  # roll/pitch/yaw
        ax.set_ylim([-np.pi, np.pi])
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for suptitle

plt.savefig("plot.png", dpi=200)
plt.show()