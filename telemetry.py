import numpy as np
from transform import *

class Telemetry:
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.body_id = model.body(name="SWIX").id
    
    def pos(self):
        return np.array([self.data.sensordata[0], self.data.sensordata[1], self.data.sensordata[2]])
    
    def rot(self):
        tfm = self.data.xmat[self.body_id].reshape(3,3)
        return quat_to_euler_zyx(self.data.sensordata[3:7])
    
    def vel(self):
        tfm = self.data.xmat[self.body_id].reshape(3,3)
        linvel = tfm.T @ np.array([self.data.sensordata[7], self.data.sensordata[8], self.data.sensordata[9]])
        angvel = tfm.T @ np.array([self.data.sensordata[10], self.data.sensordata[11], self.data.sensordata[12]])
        return np.concatenate([linvel, angvel])