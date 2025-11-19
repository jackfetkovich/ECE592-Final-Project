import mujoco
import mujoco.viewer
import time
import numpy as np


model = mujoco.MjModel.from_xml_path('project.xml')
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
data.qvel[3:6] = 5*np.random.randn(3)

with mujoco.viewer.launch_passive(model, data) as viewer:
    dt = model.opt.timestep
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)