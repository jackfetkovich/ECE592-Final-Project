import numpy as np

class Telemetry:
    def __init__(self, data):
        self.data = data
    
    def pos(self):
        return np.array([self.data[0], self.data[1], self.data[2]])
    
    def vel(self):
        return np.array([self.data[3], self.data[4], self.data[5], self.data[6], self.data[7], self.data[8]])