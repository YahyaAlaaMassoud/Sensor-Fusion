
import numpy as np
from libs.kitti.core import Box3D

class PIXORTargets:
    # def __init__(self, shape, target_means, target_stds, mean_height, mean_altitude, P_WIDTH, P_HEIGHT, P_DEPTH):
    def __init__(self, shape, P_WIDTH, P_HEIGHT, P_DEPTH, stats=None):
        self.target_height, self.target_width = shape
        # self.target_means, self.target_stds = target_means, target_stds
        # self.mean_height, self.mean_altitude = mean_height, mean_altitude
        self.stats = stats

        self.P_WIDTH, self.P_HEIGHT, self.P_DEPTH = P_WIDTH, P_HEIGHT, P_DEPTH
        
        # Quantization factor(x, y) = target dimensions / physical dimensions (Eg. 800 / 80 = 10)
        self.qf = self.target_width / P_WIDTH, self.target_height / P_HEIGHT

        xx, yy = np.meshgrid(range(0, self.target_width), range(0, self.target_height))
        self.feature_map_pts = np.vstack((xx.ravel(), yy.ravel())).T

    def get_output_shape(self):
        return self.target_height, self.target_width, 9
    
    # Generates target map for a single 3D bounding box
    def generate_offset_stats(self, box_3D):
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

        x_off, z_off = [], []
        # Generate corresponding geometry map for pts in rectangle
        for positive_pt in self.feature_map_pts[cls_map]:

            physical_z = positive_pt[0] / self.qf[0]  # Convert to physical space
            physical_x = positive_pt[1] / self.qf[1] - 40  # Convert to physical space and move RF from top to middle
            
            z_off.append(physical_z - box_3D.z)            
            x_off.append(physical_x - box_3D.x)
            
        return x_off, z_off
    
    def __normalize(self, name, value):
        return (value - self.stats['mean'][name]) / self.stats['std'][name]
    
    def __denormalize(self, name, value):
        return (value * self.stats['std'][name]) + self.stats['mean'][name]

    # Generates target map for a single 3D bounding box
    def __generate_map(self, box_3D):
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

        geometry_map = np.zeros(shape=(self.target_height, self.target_width, 8), dtype=np.float32)
        # Generate corresponding geometry map for pts in rectangle
        for positive_pt in self.feature_map_pts[cls_map]:
            # Angle
            # replacing (theta) -> (2 * theta) as proposed by BoxNet paper
            # should be values between [-1, 1]
            # geometry_map[positive_pt[1], positive_pt[0], (0, 1)] = (np.cos(box_3D.yaw), np.sin(box_3D.yaw))
            geometry_map[positive_pt[1], positive_pt[0], (0, 1)] = (self.__normalize('cos', np.cos(box_3D.yaw)), 
                                                                    self.__normalize('sin', np.sin(box_3D.yaw)))

            # Offset from center
            physical_z = positive_pt[0] / self.qf[0]  # Convert to physical space
            physical_x = positive_pt[1] / self.qf[1] - 40  # Convert to physical space and move RF from top to middle
            # geometry_map[positive_pt[1], positive_pt[0], (2, 3)] = ((physical_z - box_3D.z), (physical_x - box_3D.x))
            geometry_map[positive_pt[1], positive_pt[0], (2, 3, 4)] = (self.__normalize('dz', physical_z - box_3D.z), 
                                                                       self.__normalize('dx', physical_x - box_3D.x),
                                                                       self.__normalize('alt', box_3D.y))

            # Width and Length
            # geometry_map[positive_pt[1], positive_pt[0], (4, 5)] = (np.log(box_3D.w), np.log(box_3D.l))
            geometry_map[positive_pt[1], positive_pt[0], (5, 6, 7)] = (self.__normalize('log_w', np.log(box_3D.w)), 
                                                                       self.__normalize('log_l', np.log(box_3D.l)),
                                                                       self.__normalize('log_h', np.log(box_3D.h)))

            # # Normalize
            # geometry_map[positive_pt[1], positive_pt[0]] -= self.target_means
            # geometry_map[positive_pt[1], positive_pt[0]] /= self.target_stds

        cls_map = cls_map.astype(np.float32).reshape(self.target_height, self.target_width, 1)

        return np.concatenate([cls_map, geometry_map], axis=2)

    def encode(self, boxes_3D):
        target = np.zeros(shape=self.get_output_shape(), dtype=np.float32)
        for box_3D in boxes_3D:
            target += self.__generate_map(box_3D)
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
            z -= self.__denormalize('dz', geometry_map[positive_pt[1], positive_pt[0], 2])
            x -= self.__denormalize('dx', geometry_map[positive_pt[1], positive_pt[0], 3])
            y  = self.__denormalize('alt', geometry_map[positive_pt[1], positive_pt[0], 4])

            # Size
            # w, l = np.exp(geometry_map[positive_pt[1], positive_pt[0], (4, 5)])
            w = np.exp(self.__denormalize('log_w', geometry_map[positive_pt[1], positive_pt[0], 5]))
            l = np.exp(self.__denormalize('log_l', geometry_map[positive_pt[1], positive_pt[0], 6]))
            h = np.exp(self.__denormalize('log_h', geometry_map[positive_pt[1], positive_pt[0], 7]))

            # Angle
            yaw = np.arctan2(self.__denormalize('sin', geometry_map[positive_pt[1], positive_pt[0], 1]),
                             self.__denormalize('cos', geometry_map[positive_pt[1], positive_pt[0], 0]))

            decoded_box = Box3D(h=h, w=w, l=l,
                                x=x, y=y, z=z,
                                yaw=yaw, confidence=cls_map[positive_pt[1], positive_pt[0]],
                                cls='Car')
            boxes_3D += [decoded_box]
        return boxes_3D
    
    def __str__(self):
        return "%s - %s" % (self.__class__.__name__, self.get_output_shape())