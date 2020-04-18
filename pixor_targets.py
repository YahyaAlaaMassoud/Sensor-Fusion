
import numpy as np

from abc import ABC, abstractmethod
from core.boxes import Box3D

class TargetsEncoder(ABC):

    def __init__(self, x_min, x_max,
                       y_min, y_max,
                       z_min, z_max,
                       discretize_factor,
                       downsample_factor,
                       reg_channels,
                       subsampling,
                       stats):
        
        self.reg_channels = reg_channels
        
        self.stats = stats
        
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.z_min, self.z_max = z_min, z_max
        
        physical_dim = np.abs(x_min) + np.abs(x_max), np.abs(y_min) + np.abs(y_max), np.abs(z_min) + np.abs(z_max)
        self.physical_width, self.physical_length, self.physical_height = physical_dim
        
        cube_width, cube_length, cube_height = int(self.physical_width / discretize_factor), int(self.physical_length / discretize_factor), int(self.physical_height / discretize_factor)
        
        self.qf = np.array([cube_width  / self.physical_width, 
                            cube_length / self.physical_length, 
                            cube_height / self.physical_height])
        
        self.target_width, self.target_length = cube_width // downsample_factor, cube_length // downsample_factor
        
        if subsampling:
            self.out_channels = self.reg_channels + 2
        else:
            self.out_channels = self.reg_channels + 1
        
    @abstractmethod
    def get_output_shape(self):
        return self.target_width, self.target_length, self.reg_channels
    
    @abstractmethod
    def get_physical_shape(self):
        return self.physical_width, self.physical_length, self.physical_height
    
    @abstractmethod
    def generate_map(self, box_3d):
        pass
    
    @abstractmethod
    def encode(self, boxes_3d):
        pass
    
    @abstractmethod
    def decode(self, target, conf_thresh):
        pass
    
    def normalize(self, name, value):
        return (value - self.stats['mean'][name]) / self.stats['std'][name]
    
    def denormalize(self, name, value):
        return (value * self.stats['std'][name]) + self.stats['mean'][name]
    
class PIXORTargets:
    # def __init__(self, shape, target_means, target_stds, mean_height, mean_altitude, P_WIDTH, P_HEIGHT, P_DEPTH):
    def __init__(self, shape, P_WIDTH, P_HEIGHT, P_DEPTH, stats=None, subsampling_factor=(0.8, 1.2), num_channels=11):
        self.target_height, self.target_width = shape
        # self.target_means, self.target_stds = target_means, target_stds
        # self.mean_height, self.mean_altitude = mean_height, mean_altitude
        self.stats = stats

        self.P_WIDTH, self.P_HEIGHT, self.P_DEPTH = P_WIDTH, P_HEIGHT, P_DEPTH
        
        # Quantization factor(x, y) = target dimensions / physical dimensions (Eg. 800 / 80 = 10)
        self.qf = self.target_width / P_WIDTH, self.target_height / P_HEIGHT

        self.subsampling_factor = subsampling_factor

        xx, yy = np.meshgrid(range(0, self.target_width), range(0, self.target_height))
        self.feature_map_pts = np.vstack((xx.ravel(), yy.ravel())).T
        
        self.num_channels = num_channels

    def get_output_shape(self):
        return self.target_height, self.target_width, self.num_channels + 1

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

    def __get_positive_pts_mask(self, box):
        # Get 4 corners of the rectangle in BEV and transform to feature coordinate frame
        corners = np.array(box.get_corners().T[0:4, [2, 0]])  # Shape: 4(#corners), 2 (#coordinates/z,x)
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
        return ((0 <= proj_0P_on_01) & (proj_0P_on_01 <= mag_01) & (0 <= proj_1P_on_12) & (proj_1P_on_12 <= mag_12))

    # Generates target map for a single 3D bounding box
    def __generate_map(self, target_box):
        inner_box = Box3D(h=target_box.h, w=target_box.w * self.subsampling_factor[0], l=target_box.l * self.subsampling_factor[0],
                          x=target_box.x, y=target_box.y, z=target_box.z, yaw=target_box.yaw, cls=target_box.cls)

        outer_box = Box3D(h=target_box.h, w=target_box.w * self.subsampling_factor[1], l=target_box.l * self.subsampling_factor[1],
                          x=target_box.x, y=target_box.y, z=target_box.z, yaw=target_box.yaw, cls=target_box.cls)

        # Obj mask - (0 - ignore, 1 - consider in loss)
        inner_mask = self.__get_positive_pts_mask(inner_box)
        outer_mask = self.__get_positive_pts_mask(outer_box)
        obj_mask = np.logical_not(np.logical_xor(inner_mask, outer_mask))

        # Generate corresponding geometry map for pts in rectangle
        # cos, sin, z, x, width, length, y, h
        # y, h - are not regressed in original paper

        # Geo mask (0 - ignore, 1 - positive)
        geo_mask = self.__get_positive_pts_mask(target_box)


        geometry_map = np.zeros(shape=(self.target_height, self.target_width, self.num_channels), dtype=np.float32)
        # Generate corresponding geometry map for pts in rectangle
        for positive_pt in self.feature_map_pts[geo_mask]:
            # Angle
            # replacing (theta) -> (2 * theta) as proposed by BoxNet paper
            # should be values between [-1, 1]
            # geometry_map[positive_pt[1], positive_pt[0], (0, 1)] = (np.cos(box_3D.yaw), np.sin(box_3D.yaw))
            geometry_map[positive_pt[1], positive_pt[0], (0, 1)] = (self.__normalize('cos', np.cos(target_box.yaw)), 
                                                                    self.__normalize('sin', np.sin(target_box.yaw)))

            # Offset from center
            physical_z = positive_pt[0] / self.qf[0]  # Convert to physical space
            physical_x = positive_pt[1] / self.qf[1] - 40  # Convert to physical space and move RF from top to middle
            # geometry_map[positive_pt[1], positive_pt[0], (2, 3)] = ((physical_z - box_3D.z), (physical_x - box_3D.x))
            geometry_map[positive_pt[1], positive_pt[0], (2, 3, 4)] = (self.__normalize('dz', physical_z - target_box.z), 
                                                                       self.__normalize('dx', physical_x - target_box.x),
                                                                       self.__normalize('alt', target_box.y))

            # # Width and Length
            # # geometry_map[positive_pt[1], positive_pt[0], (4, 5)] = (np.log(box_3D.w), np.log(box_3D.l))
            geometry_map[positive_pt[1], positive_pt[0], (5, 6, 7)] = (self.__normalize('log_w', np.log(target_box.w)), 
                                                                       self.__normalize('log_l', np.log(target_box.l)),
                                                                       self.__normalize('log_h', np.log(target_box.h)))

            # scaling ratio experiment
            geometry_map[positive_pt[1], positive_pt[0], (8, 9,  10)] = (np.log(target_box.w) / self.stats['mean']['log_w'], 
                                                                        np.log(target_box.l) / self.stats['mean']['log_l'],
                                                                        np.log(target_box.h) / self.stats['mean']['log_h'],)

            # # Normalize
            # geometry_map[positive_pt[1], positive_pt[0]] -= self.target_means
            # geometry_map[positive_pt[1], positive_pt[0]] /= self.target_stds

        # Reshape flat masks into 2D array
        obj_map = inner_mask.astype(np.float32).reshape(self.target_height, self.target_width)
        obj_mask = obj_mask.astype(np.float32).reshape(self.target_height, self.target_width)
        geo_mask = geo_mask.astype(np.float32).reshape(self.target_height, self.target_width)

        return obj_map, obj_mask, geometry_map, geo_mask

    def encode(self, boxes_3D):
        # Obj_map contains map and mask
        obj_map = np.zeros(shape=(self.target_height, self.target_width, 2), dtype=np.float32)  # Channels   0 - label   1 - mask
        obj_map[..., 1] = 1
        geo_map = np.zeros(shape=(self.target_height, self.target_width, self.num_channels + 1), dtype=np.float32)  # Channels   0-10 - regression vars   11 - mask
        for box_3D in boxes_3D:
            _obj_map, _obj_mask, _geo_map, _geo_mask = self.__generate_map(box_3D)
            obj_map[..., 0] += _obj_map
            obj_map[..., 1] = np.multiply(obj_map[..., 1], _obj_mask)
            geo_map[..., :-1] += _geo_map
            geo_map[..., -1] += _geo_mask

        return obj_map, geo_map

    def encode_batch(self, boxes_3D_batch):
        batch_size = len(boxes_3D_batch)
        obj_map = np.zeros(shape=(batch_size, self.target_height, self.target_width, 2), dtype=np.float32)  # Channels   0 - label   1 - mask
        geo_map = np.zeros(shape=(batch_size, self.target_height, self.target_width, self.num_channels + 1), dtype=np.float32)  # Channels   0-7 - regression vars   8 - mask

        for i in range(batch_size):
            obj_map[i], geo_map[i] = self.encode(boxes_3D_batch[i])

        return obj_map, geo_map

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

            # # Size
            # scaling ratio experiment
            w1 = np.exp(geometry_map[positive_pt[1], positive_pt[0], 8] * self.stats['mean']['log_w'])
            l1 = np.exp(geometry_map[positive_pt[1], positive_pt[0], 9] * self.stats['mean']['log_l'])
            h1 = np.exp(geometry_map[positive_pt[1], positive_pt[0], 10] * self.stats['mean']['log_h'])

            ap = np.log(w1) * np.log(l1)
            at = self.stats['mean']['log_w'] * self.stats['mean']['log_l']
            if ap < at / 3:
                continue

            # # # Size
            # # # w, l = np.exp(geometry_map[positive_pt[1], positive_pt[0], (4, 5)])
            w = np.exp(self.__denormalize('log_w', geometry_map[positive_pt[1], positive_pt[0], 5]))
            l = np.exp(self.__denormalize('log_l', geometry_map[positive_pt[1], positive_pt[0], 6]))
            h = np.exp(self.__denormalize('log_h', geometry_map[positive_pt[1], positive_pt[0], 7]))

            w = np.mean([w1, w])#w1#
            l = np.mean([l1, l])#l1#
            h = np.mean([h1, h])#h1#

            # Angle
            yaw = np.arctan2(self.__denormalize('sin', geometry_map[positive_pt[1], positive_pt[0], 1]),
                             self.__denormalize('cos', geometry_map[positive_pt[1], positive_pt[0], 0]))

            decoded_box = Box3D(h=h, w=w, l=l,
                                x=x, y=y, z=z,
                                yaw=yaw, confidence=cls_map[positive_pt[1], positive_pt[0]],
                                cls='Car')
            boxes_3D.append(decoded_box)
        return boxes_3D
    
    def __str__(self):
        return "%s - %s" % (self.__class__.__name__, self.get_output_shape())