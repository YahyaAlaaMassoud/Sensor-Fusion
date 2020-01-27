
import numpy as np

class OccupancyCuboid:
    def __init__(self, shape, P_WIDTH, P_HEIGHT, P_DEPTH):
        self.cube_height, self.cube_width, self.cube_depth = shape
        self.P_WIDTH, self.P_HEIGHT, self.P_DEPTH = P_WIDTH, P_HEIGHT, P_DEPTH

        # Quantization factor(x, y, z) = cube dimensions / physical dimension  (Eg. 800 / 80 = 10)
        self.qf = np.array([self.cube_width / P_WIDTH, self.cube_height / P_HEIGHT, self.cube_depth / P_DEPTH])

    def get_output_shape(self):
        return self.cube_height, self.cube_width, self.cube_depth

    # Fully vectorized implementation
    def encode(self, pts, reflectance):
        # pts = self.__add_pts(pts, -1., 0.15)
        # pts are in physical space; (ix, iy, iz) are in cube space
        ix = ((pts[:, 2]) * self.qf[0]).astype(np.int32)
        iy = ((pts[:, 0] + self.P_HEIGHT / 2) * self.qf[1]).astype(np.int32)
        iz = ((pts[:, 1] + 1) * self.qf[2]).astype(np.int32)

        occupancy_grid = np.zeros(shape=self.get_output_shape(), dtype=np.float32)

        # ones = []
        # for i in iz:
        #     arr = np.zeros((self.cube_depth))
        #     arr[i:] = 1.
        #     ones.append(arr)
        # ones = np.array(ones)
        # occupancy_grid[iy, ix] = ones
        occupancy_grid[iy, ix, iz] = 1.
        
        return occupancy_grid

    def __add_pts(self, pts, z_min, every):
        new_pts = []
        for i in range(len(pts)):
            x, y, z = pts[i,0], pts[i,1], pts[i,2]
            while y >= z_min:
                y -= every
                new_pts.append(np.array([x,y,z]))
        return np.array(new_pts)
    
    def __str__(self):
        return "%s - %s" % (self.__class__.__name__, self.get_output_shape())