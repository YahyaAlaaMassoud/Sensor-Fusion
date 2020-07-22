
import numpy as np

class BEVVoxelizer(object):
    def __init__(self, n_x, n_y, n_z, side_range, fwd_range, height_range, filter_external=True):
        """Grid of voxels with support for different build methods.
        Parameters
        ----------
        n_x, n_y, n_z :  int
            The number of segments in which each axis will be divided.
            Ignored if corresponding size_x, size_y or size_z is not None.
        side_range, fwd_range, height_range: tuples (min, max) define the walls of the occupancy grid
        filter_external: boolean
            Whether or not to remove points from the cloud or have them accumulate at the edges
        """
        
        self.x_y_z = [n_x, n_y, n_z]
        self.xyzmin = [side_range[0], fwd_range[0], height_range[0]]
        self.xyzmax = [side_range[1], fwd_range[1], height_range[1]]
        self.filter_external = filter_external
        self.side_range = side_range
        self.fwd_range = fwd_range
        self.height_range = height_range

    def encode_batch(self, pc_batch):
        return np.array([self.encode(pc) for pc in pc_batch])

    def encode(self, pts):
        '''pts: (N, 3) numpy.array
        '''
        
        if pts.shape[1] != 3:
            pts = pts.T


        '''
            make sure axes are well arranged (should not be placed here!!)
        '''
        new_pc = np.zeros(pts.shape)
        new_pc[:,0] = pts[:,0]
        new_pc[:,1] = pts[:,2]
        new_pc[:,2] = pts[:,1]

        pts = new_pc
        
        if self.filter_external:
            # FILTER - To return only indices of points within desired box
            # Three filters for: Front-to-back (y), side-to-side (x), and height (z) ranges
            # Note left side is positive y axis in LIDAR coordinates
            f_filt = np.logical_and((pts[:,0] > self.fwd_range[0]), (pts[:,0] < self.fwd_range[1]))
            s_filt = np.logical_and((pts[:,1] > self.side_range[0]), (pts[:,1] < self.side_range[1]))
            h_filt = np.logical_and((pts[:,2] > self.height_range[0]), (pts[:,2] < self.height_range[1]))
            filter_xy = np.logical_and(f_filt, s_filt)
            filter = np.logical_and(filter_xy, h_filt)
            indices = np.argwhere(filter).flatten()
            self.pts = pts[indices,:]
        
        else:
            self.pts = pts
        
        segments = []
        shape = []

        for i in range(3):
            # note the +1 in num
            s, step = np.linspace(self.xyzmin[i], self.xyzmax[i], num=self.x_y_z[i]+1, retstep=True)
            segments.append(s)
            shape.append(step)

        self.segments = segments
        self.shape = shape

        self.n_voxels = self.x_y_z[0] * self.x_y_z[1] * self.x_y_z[2]

        # find where each point lies in corresponding segmented axis
        self.voxel_x = np.clip(segments[0].size - np.searchsorted(segments[0],  self.pts[:,1], side = "right") - 1, 0, self.x_y_z[0] - 1)
        self.voxel_y = np.clip(segments[1].size - np.searchsorted(segments[1],  self.pts[:,0],  side = "right") - 1, 0, self.x_y_z[1] - 1)
        self.voxel_z = np.clip(segments[2].size - np.searchsorted(segments[2],  self.pts[:,2],  side = "right") - 1, 0, self.x_y_z[2] - 1)
        self.voxel_n = np.ravel_multi_index([self.voxel_x, self.voxel_y, self.voxel_z], self.x_y_z)

        # compute center of each voxel
        self.midsegments = [(self.segments[i][1:] + self.segments[i][:-1]) / 2 for i in range(3)]
        self.voxel_centers = cartesian(self.midsegments).astype(np.float32)

        grid = np.zeros((self.x_y_z[1],self.x_y_z[0],self.x_y_z[2]), dtype=np.float32)
        grid[self.voxel_y,self.voxel_x,self.voxel_z] = 1.
        grid = np.rot90(grid, -1)

        return grid

def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out