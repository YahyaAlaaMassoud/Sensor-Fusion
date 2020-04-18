
import deepdish as dd
import numpy as np

from pixor_targets import TargetsEncoder

class PixorTargets3D(TargetsEncoder):
    
    def __init__(self, x_min, x_max,
                       y_min, y_max,
                       z_min, z_max,
                       discretize_factor,
                       downsample_factor,
                       subsampling,
                       stats):
        
        self.reg_channels = 8
        
        super().__init__(x_min, x_max, y_min, y_max, z_min, z_max, discretize_factor, downsample_factor,
                         self.reg_channels, subsampling, stats)
        
    def get_output_shape(self):
        return self.target_length, self.target_width, self.out_channels
    
    def get_physical_shape(self):
        return self.physical_length, self.physical_width, self.physical_height
    
    def generate_map(self, box_3D):
        # Get 4 corners of the rectangle in BEV and transform to feature coordinate frame
        corners = np.array(box_3D.get_corners().T[0:4, [2, 0]])  # Shape: 4(#corners), 2 (#coordinates/z,x)
        corners[:, 1] += 40  # Move RF to the top
        # Convert from physical space (80 x 70m) to (200 x 175px) feature space
        # TODO: Convert to oneliner
        corners[:, 0] = self.qf[0] * corners[:, 0]
        corners[:, 1] = self.qf[1] * corners[:, 1]

        # Compute class map (Objectness)
        V01, V12 = corners[1] - corners[0], corners[2] - corners[1]
        V0P, V1P = self.feature_map_pts - corners[0], self.feature_map_pts - corners[1]
        proj_0P_on_01, proj_1P_on_12 = np.dot(V0P, V01), np.dot(V1P, V12)
        mag_01, mag_12 = np.dot(V01, V01), np.dot(V12, V12)
        cls_map = ((0 <= proj_0P_on_01) & (proj_0P_on_01 <= mag_01) & (0 <= proj_1P_on_12) & (proj_1P_on_12 <= mag_12))

        geometry_map = np.zeros(shape=(self.target_length, self.target_width, 8), dtype=np.float32)
        # Generate corresponding geometry map for pts in rectangle
        for positive_pt in self.feature_map_pts[cls_map]:
            # Angle
            # replacing (theta) -> (2 * theta) as proposed by BoxNet paper
            # should be values between [-1, 1]
            # geometry_map[positive_pt[1], positive_pt[0], (0, 1)] = (np.cos(box_3D.yaw), np.sin(box_3D.yaw))
            geometry_map[positive_pt[1], positive_pt[0], (0, 1)] = (self.normalize('cos', np.cos(box_3D.yaw)), 
                                                                    self.normalize('sin', np.sin(box_3D.yaw)))

            # Offset from center
            physical_z = positive_pt[0] / self.qf[0]  # Convert to physical space
            physical_x = positive_pt[1] / self.qf[1] - 40  # Convert to physical space and move RF from top to middle
            # geometry_map[positive_pt[1], positive_pt[0], (2, 3)] = ((physical_z - box_3D.z), (physical_x - box_3D.x))
            geometry_map[positive_pt[1], positive_pt[0], (2, 3, 4)] = (self.normalize('dz', physical_z - box_3D.z), 
                                                                       self.normalize('dx', physical_x - box_3D.x),
                                                                       self.normalize('alt', box_3D.y))

            # Width and Length
            # geometry_map[positive_pt[1], positive_pt[0], (4, 5)] = (np.log(box_3D.w), np.log(box_3D.l))
            geometry_map[positive_pt[1], positive_pt[0], (5, 6, 7)] = (self.normalize('log_w', np.log(box_3D.w)), 
                                                                       self.normalize('log_l', np.log(box_3D.l)),
                                                                       self.normalize('log_h', np.log(box_3D.h)))

            # # Normalize
            # geometry_map[positive_pt[1], positive_pt[0]] -= self.target_means
            # geometry_map[positive_pt[1], positive_pt[0]] /= self.target_stds

        cls_map = cls_map.astype(np.float32).reshape(self.target_length, self.target_width, 1)

        return np.concatenate([cls_map, geometry_map], axis=2)
    
    def encode(self, boxes_3d):
        xx, yy = np.meshgrid(range(0, self.target_width), range(0, self.target_length))
        self.feature_map_pts = np.vstack((xx.ravel(), yy.ravel())).T
        
        target = np.zeros(shape=self.get_output_shape(),
                          dtype=np.float32)
        
        for box in boxes_3d:
            target += self.generate_map(box)
            
        return target
    
    def decode(self, target, confidence_thresh):
        boxes_3D = []
        cls_map = target[..., 0]
        cls_map_flat = cls_map.flatten()
        geometry_map = target[..., 1:]
        for positive_pt in self.feature_map_pts[cls_map_flat > confidence_thresh]:
            # Denormalize
            # geometry_map[positive_pt[1], positive_pt[0]] *= self.target_stds
            # geometry_map[positive_pt[1], positive_pt[0]] += self.target_means

            # Offset
            z = positive_pt[0] / self.qf[0]  # Convert to physical space
            x = positive_pt[1] / self.qf[1] - self.P_HEIGHT / 2  # Convert to physical space and move reference frame from top to middle
            z -= self.denormalize('dz', geometry_map[positive_pt[1], positive_pt[0], 2])
            x -= self.denormalize('dx', geometry_map[positive_pt[1], positive_pt[0], 3])
            y  = self.denormalize('alt', geometry_map[positive_pt[1], positive_pt[0], 4])

            # Size
            # w, l = np.exp(geometry_map[positive_pt[1], positive_pt[0], (4, 5)])
            w = np.exp(self.denormalize('log_w', geometry_map[positive_pt[1], positive_pt[0], 5]))
            l = np.exp(self.denormalize('log_l', geometry_map[positive_pt[1], positive_pt[0], 6]))
            h = np.exp(self.denormalize('log_h', geometry_map[positive_pt[1], positive_pt[0], 7]))

            # Angle
            yaw = np.arctan2(self.denormalize('sin', geometry_map[positive_pt[1], positive_pt[0], 1]),
                             self.denormalize('cos', geometry_map[positive_pt[1], positive_pt[0], 0]))

            decoded_box = Box3D(h=h, w=w, l=l,
                                x=x, y=y, z=z,
                                yaw=yaw, confidence=cls_map[positive_pt[1], positive_pt[0]],
                                cls='Car')
            boxes_3D += [decoded_box]
        return boxes_3D
    
# pt = PixorTargets3D(x_min=0, x_max=70, y_min=-40, y_max=40, z_min=-1, z_max=2.5, subsampling=False,
#                     discretize_factor=0.1, downsample_factor=4, stats=dd.io.load('kitti_stats/stats.h5'))
# print(pt.get_physical_shape())
# print(pt.get_output_shape())
# pt.encode([])