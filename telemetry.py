import numpy as np
from transform import *

class Telemetry:
    def __init__(self, data):
        self.data = data
    
    def pos(self):
        return np.array([self.data.sensordata[0], self.data.sensordata[1], self.data.sensordata[2]])
    
    def rot(self):
        return quat_to_euler_zyx(self.data.sensordata[3:6])
    
    def vel(self):
        return np.array([self.data.sensordata[6], self.data.sensordata[7], self.data.sensordata[8], self.data.sensordata[9], self.data.sensordata[10], self.data.sensordata[11]])