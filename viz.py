"""
Copyright 2019-2020 Selameab (https://github.com/Selameab)
"""

import os

import cv2
import matplotlib
import numpy as np
import open3d as o3d

from core import transforms_3D, transforms_2D
from core.boxes import Box2D, Box3D, get_corners_3D

matplotlib.use('pdf')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

GT_COLORS = {
    'Car': 'green',
    'Van': 'blue',
    'Truck': 'darkcyan',
    'Pedestrian': 'peru',
    'Person_sitting': 'goldenrod',
    'Cyclist': 'orange',
    'Tram': 'darkmagenta',
    'Misc': 'indigo',
    'DontCare': 'orchid',
}

# PRED_COLORS are lighter versions of GT_COLORS
PRED_COLORS = {}
for i in GT_COLORS:
    GT_COLORS[i] = np.array(mcolors.to_rgba(GT_COLORS[i]))
    _color = np.clip(GT_COLORS[i] + 0.28, a_min=0, a_max=1)
    _color[-1] = 1.0
    PRED_COLORS[i] = _color

DEFAULT_COLOR = [0.8, 0.8, 1.0]

BOX_CONNECTIONS_3D = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3], [1, 6], [2, 5]]
BOX_CONNECTIONS_2D = [[0, 1], [1, 2], [2, 3], [3, 0]]
LINE_CONNECTIONS_3D = [[0, 1]]


def imsave(img, path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img * 255.0)


def imshow(img, scale=15, block=False):
    fig, ax = plt.subplots()
    # ar = img.shape[0] / img.shape[1]  # aspect_ratio = h / w
    # fig.set_size_inches(scale, scale * ar)

    ax.imshow(img)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # fig.tight_layout()
    plt.show()


def range_view(img, P2=None, gt_boxes=None, pred_boxes=None):
    img = np.copy(img)  # Clone

    def draw_boxes_2D(boxes, color_dict):
        for box in boxes:
            cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color_dict.get(box.cls, DEFAULT_COLOR), 2)

    def draw_boxes_3D(boxes, color_dict):
        for box in boxes:
            corners = transforms_3D.project(P2, get_corners_3D(box)).astype(np.int32)
            for start, end in BOX_CONNECTIONS_3D:
                x1, y1 = corners[:, start]
                x2, y2 = corners[:, end]
                cv2.line(img, (x1, y1), (x2, y2), color_dict.get(box.cls, DEFAULT_COLOR), 1)

    if gt_boxes is not None and len(gt_boxes) > 0:
        if isinstance(gt_boxes[0], Box2D):
            draw_boxes_2D(gt_boxes, GT_COLORS)
        elif isinstance(gt_boxes[0], Box3D):
            assert P2 is not None
            draw_boxes_3D(gt_boxes, GT_COLORS)

    if pred_boxes is not None and len(pred_boxes) > 0:
        if isinstance(pred_boxes[0], Box2D):
            draw_boxes_2D(pred_boxes, PRED_COLORS)
        elif isinstance(pred_boxes[0], Box3D):
            assert P2 is not None
            draw_boxes_3D(pred_boxes, PRED_COLORS)

    return img


def bev(pts=None, gt_boxes=None, pred_boxes=None, scale=10, title=None, circles=None):
    img_shape = int(80 * scale), int(70 * scale)
    canvas = np.ones(img_shape + (3,), dtype=np.float32) * 0.05

    # Transforms from physical space to canvas
    H = np.dot(transforms_2D.scale_matrix(img_shape[0] / 80, img_shape[1] / 70),
               transforms_2D.translation_matrix(0, 40))

    # pts
    if pts is not None:
        pts = pts[[2, 0]]  # Extract x and z coordinates
        pts = transforms_2D.transform(H, pts)
        pts = pts.astype(np.int)
        canvas[pts[1], pts[0]] = 1

    def draw_boxes(boxes, color_dict, pred=False):
        for box in boxes:
            color = color_dict.get(box.cls, DEFAULT_COLOR)
            if pred:
                color = [1., 0., 0.]
            else:
                color = [0., 1., 0.]
            corners = box.get_bev_box()
            corners = transforms_2D.transform(H, corners)
            for start, end in BOX_CONNECTIONS_2D:
                x1, y1 = corners[:, start]
                x2, y2 = corners[:, end]
                cv2.line(canvas, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

            # Draw arrow from center to front face
            cx, cy = np.mean(corners, axis=1)
            ax, ay = np.mean(corners[:, 1:3], axis=1)
            cv2.line(canvas, (cx, cy), (ax, ay), color, 1, cv2.LINE_AA)
            
            if circles is not None:
                for circle, clr in circles:
                    circle = transforms_2D.transform(H, np.expand_dims(np.array([circle[0], circle[1]]), axis=1))
                    circle = np.squeeze(circle)
                    cv2.circle(canvas, (int(circle[0]), int(circle[1])), 3, clr, -1)
                    if clr == (0, 255, 0):
                        cv2.line(canvas, (0, 400), (int(circle[0]), int(circle[1])), clr, 1, cv2.LINE_AA)

            # Write text
            if box.text is not None and len(box.text) > 0:
                cv2.putText(canvas, box.text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, box.text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)

    # boxes
    if gt_boxes is not None:
        draw_boxes(gt_boxes, GT_COLORS, pred=False)
    if pred_boxes is not None:
        draw_boxes(pred_boxes, PRED_COLORS, pred=True)

    # Title
    if title is not None and len(title) > 0:
        cv2.putText(canvas, title, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0.3, 0.8, 0.4), 1, cv2.LINE_AA)

    return canvas


def __pts_to_line_set(pts, connections, color):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts.T)
    line_set.lines = o3d.utility.Vector2iVector(connections)
    line_set.colors = o3d.utility.Vector3dVector([mcolors.to_rgba(color)[:3] for _ in range(len(connections))])
    return line_set


def __create_sphere(x, y, z, r, color):
    sphere = o3d.geometry.create_mesh_sphere(r)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(mcolors.to_rgba(color)[:3])
    sphere.transform(np.array([[1, 0, 0, x],
                               [0, 1, 0, y],
                               [0, 0, 1, z],
                               [0, 0, 0, 1]]))


def open3d(pts=None, gt_boxes=None, sampled_boxes=None, pred_boxes=None, limits=None):
    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=600)

    # Create coordinate frame at sensor(camera)
    camera_rf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(camera_rf)

    # Draw GT Boxes
    if gt_boxes is not None:
        for box in gt_boxes:
            vis.add_geometry(__pts_to_line_set(pts=get_corners_3D(box), connections=BOX_CONNECTIONS_3D, color=GT_COLORS[box.cls]))

    if sampled_boxes is not None:
        for box in sampled_boxes:
            vis.add_geometry(__pts_to_line_set(pts=get_corners_3D(box), connections=BOX_CONNECTIONS_3D, color='yellow'))

    # Draw pred_boxes
    if pred_boxes is not None:
        for box in pred_boxes:
            vis.add_geometry(__pts_to_line_set(pts=get_corners_3D(box), connections=BOX_CONNECTIONS_3D, color=PRED_COLORS[box.cls]))
            
    if limits is not None:
        for limit in limits:
            vis.add_geometry(__pts_to_line_set(pts=limit.T, connections=LINE_CONNECTIONS_3D, color='red'))

    # Create point cloud
    if pts is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.T)
        vis.add_geometry(pcd)

    # Change background to black
    vis.get_render_option().background_color = np.array([0, 0, 0])

    # Load camera params (view angle, etc)
    camera_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'open3d_config', 'camera_params.json')
    vis.get_view_control().convert_from_pinhole_camera_parameters(o3d.io.read_pinhole_camera_parameters(camera_params_path))

    # Load render options
    render_options_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'open3d_config', 'render_options.json')
    vis.get_render_option().load_from_json(render_options_path)

    vis.run()
    vis.destroy_window()


def main():
    from core.kitti import KITTI, ALL_OBJECTS, CARS_ONLY
    DS_DIR = '/home/salam/datasets/KITTI/training'
    reader = KITTI(DS_DIR, CARS_ONLY)
    ids = reader.get_ids('train')
    for t in [ids[4]]:
        img = reader.get_image(t)
        pts, ref = reader.get_velo(t, use_fov_filter=False)
        boxes_2D = reader.get_boxes_2D(t)
        boxes_3D = reader.get_boxes_3D(t)
        P2 = reader.get_calib(t)[2]

        for b in boxes_3D:
            b.text = f"{np.random.uniform(0.0, 1.0, (1,))[0]:0.2f}"

        imshow(bev(pts, pred_boxes=boxes_3D, title="GT"))
        # imshow(range_view(img, P2, boxes_2D))
        # open3d(pts, boxes_3D)


if __name__ == '__main__':
    main()
