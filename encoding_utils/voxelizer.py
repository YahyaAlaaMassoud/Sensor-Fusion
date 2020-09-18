
import numpy as np
import timeit

from numba import jit, extending
from voxelizer_encoder import vox_encoder

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
        self.side_range = list(side_range)
        self.fwd_range = list(fwd_range)
        self.height_range = list(height_range)

    def encode_batch(self, pc_batch):
        return np.array([self.encode(pc) for pc in pc_batch])

    def encode(self, pts):
        return _encode(pts, 
                       self.x_y_z, 
                       self.xyzmin,
                       self.xyzmax,
                       self.filter_external,
                       self.side_range,
                       self.fwd_range,
                       self.height_range)


# @extending.overload(np.clip)
# def np_clip(a, a_min, a_max, out=None):
#     def np_clip_impl(a, a_min, a_max, out=None):
#         if out is None:
#             out = np.empty_like(a)
#         for i in range(len(a)):
#             if a[i] < a_min:
#                 out[i] = a_min
#             elif a[i] > a_max:
#                 out[i] = a_max
#             else:
#                 out[i] = a[i]
#         return out
#     return np_clip_impl


def _encode(pts, x_y_z, xyzmin, xyzmax, filter_external, side_range, fwd_range, height_range):
    # return vox_encoder(pts, x_y_z, xyzmin, xyzmax, filter_external, side_range, fwd_range, height_range) # Pythran

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
    
    if filter_external:
        # FILTER - To return only indices of points within desired box
        # Three filters for: Front-to-back (y), side-to-side (x), and height (z) ranges
        # Note left side is positive y axis in LIDAR coordinates
        f_filt = np.logical_and((pts[:,0] > fwd_range[0]), (pts[:,0] < fwd_range[1]))
        s_filt = np.logical_and((pts[:,1] > side_range[0]), (pts[:,1] < side_range[1]))
        h_filt = np.logical_and((pts[:,2] > height_range[0]), (pts[:,2] < height_range[1]))
        filter_xy = np.logical_and(f_filt, s_filt)
        filter = np.logical_and(filter_xy, h_filt).astype(np.int64)
        filter = np.nonzero(filter)[0]
        indices = np.transpose(filter).flatten() # np.argwhere(filter).flatten()
        pts = pts[indices,:]
    
    else:
        pts = pts
    
    segments = []
    # shape = []

    for i in range(3):
        # note the +1 in num
        # s, step = np.linspace(xyzmin[i], xyzmax[i], num=x_y_z[i]+1, retstep=True)
        s = np.linspace(xyzmin[i], xyzmax[i], x_y_z[i]+1)
        segments.append(s)
        # shape.append(step)

    # n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]

    # find where each point lies in corresponding segmented axis
    voxel_x = np.clip(segments[0].size - np.searchsorted(segments[0], pts[:,1], side = "right") - 1, 0, x_y_z[0] - 1).astype(np.int64)
    voxel_y = np.clip(segments[1].size - np.searchsorted(segments[1], pts[:,0],  side = "right") - 1, 0, x_y_z[1] - 1).astype(np.int64)
    voxel_z = np.clip(segments[2].size - np.searchsorted(segments[2], pts[:,2],  side = "right") - 1, 0, x_y_z[2] - 1).astype(np.int64)
    # voxel_n = np.ravel_multi_index([voxel_x, voxel_y, voxel_z], x_y_z)

    # compute center of each voxel
    # midsegments = [(segments[i][1:] + segments[i][:-1]) / 2 for i in range(3)]
    # voxel_centers = cartesian(midsegments).astype(np.float32)

    grid = np.zeros((x_y_z[1], x_y_z[0], x_y_z[2])).astype(np.float64)
    # for y, x, z in zip(voxel_y, voxel_x, voxel_z):
    #     grid[y, x, z] = 1.
    grid[voxel_y, voxel_x, voxel_z] = 1.
    grid = np.rot90(grid, -1)

    return grid


#pythran export cartesian(float list)
def cartesian(arrays):
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
    shape = [len(x) for x in arrays]
    dtype = arrays[0].dtype

    # print(shape)
    ix = np.indices(shape)
    # print(ix.shape)
    ix = ix.reshape(len(arrays), -1).T
    # print(ix.shape)

    out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


# @jit(nopython=True)
def indices(dimensions, dtype=int, sparse=False):
    N = len(dimensions)
    shape = [1,]*N
    if sparse:
        res = tuple()
    else:
        res = np.empty([N,]+dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        idx = np.arange(dim, dtype=dtype).reshape(
            shape[:i] + [dim,] + shape[i+1:]
        )
        if sparse:
            res = res + [idx,]
        else:
            res[i] = idx
    return res