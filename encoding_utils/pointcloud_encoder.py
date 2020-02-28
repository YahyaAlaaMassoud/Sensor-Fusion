
import numpy as np

from abc import ABC, abstractmethod

class PointCloudEncoder(ABC):
    
    def __init__(self, x_min, x_max,
                       y_min, y_max,
                       z_min, z_max,
                       df, densify=False):
        
        self.densify = densify
        
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.z_min, self.z_max = z_min, z_max
        
        physical_dim = np.abs(x_min) + np.abs(x_max), np.abs(y_min) + np.abs(y_max), np.abs(z_min) + np.abs(z_max)
        self.physical_width, self.physical_length, self.physical_height = physical_dim
        
        self.cube_width, self.cube_length, self.cube_height = int(self.physical_width / df), int(self.physical_length / df), int(self.physical_height / df)
        
        self.qf = np.array([self.cube_width / self.physical_width, 
                            self.cube_length / self.physical_length, 
                            self.cube_height / self.physical_height])
        
    @abstractmethod
    def get_output_shape(self):
        return self.cube_width, self.cube_length, self.cube_height
    
    @abstractmethod
    def get_physical_shape(self):
        return self.physical_width, self.physical_length, self.physical_height
    
    @abstractmethod
    def encode(self, pts, ref):
        pass

class OccupancyCuboidKITTI(PointCloudEncoder):
    
    # Fully vectorized implementation
    def encode(self, pts, reflectance=None):
        
        ix = ((pts[:, 2] + (-1. * self.x_min)) * self.qf[0]).astype(np.int32)
        iy = ((pts[:, 0] + (-1. * self.y_min)) * self.qf[1]).astype(np.int32)
        iz = ((pts[:, 1] + (-1. * self.z_min)) * self.qf[2]).astype(np.int32)
        ix[ix >= 700] = 699
        iy[iy >= 800] = 799
        iz[iz >= 35] = 34

        occupancy_grid = np.zeros(shape=self.get_output_shape(), dtype=np.float32)

        if self.densify:
            ones = []
            for i in iz:
                arr = np.zeros((self.cube_height))
                arr[:i] = 1.
                ones.append(arr)
            occupancy_grid[iy, ix] = np.array(ones)
        else:
            occupancy_grid[iy, ix, iz] = 1.
        
        return occupancy_grid
    
    def get_output_shape(self):
        return self.cube_length, self.cube_width, self.cube_height
    
    def get_physical_shape(self):
        return self.physical_length, self.physical_width, self.physical_height
    
# s = OccupancyCuboidKITTI(x_min=0, x_max=70, y_min=-40, y_max=40, z_min=-1, z_max=2.5, df=0.1)
# print(s.get_output_shape())
# print(s.get_physical_shape())