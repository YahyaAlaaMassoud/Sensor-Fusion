import os, sys

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.widgets import Slider
from multiprocessing import Process

from .core import project
from .point_counter import compute_mask_accurate

# Constants
BOX_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3]]
ARROW_CONNECTIONS = [[0, 1]]

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
    None: 'orchid'
}

# PRED_COLORS are lighter versions of GT_COLORS
PRED_COLORS = {}
for k, v in GT_COLORS.items():
    color = np.clip(np.array(mcolors.to_rgba(v)) + 0.28, a_min=0, a_max=1)
    color[-1] = 1.0
    PRED_COLORS[k] = color


# Only used to remove extreme boxes, not accurate representation of actual workspace
def __is_box_in_workspace(box):
    return True
    return (-50 < box.x < 50) and (-5 < box.y < 5) and (-20 < box.z < 100)


def view_2D(img, gt_boxes=None, pred_boxes=None, scale=1, title="", ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Camera View with 2D Boxes - ' + title)
        fig.set_size_inches(scale * 7, scale * 8)

    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Draw GT boxes
    if gt_boxes is not None:
        for gt_box in gt_boxes:
            rect = mpatches.Rectangle((gt_box.x1, gt_box.y1), gt_box.w, gt_box.h,
                                      linewidth=1, fill=None, edgecolor=GT_COLORS[gt_box.cls], linestyle='--')
            ax.add_patch(rect)

    # Draw predicted boxes
    if pred_boxes is not None:
        for pred_box in pred_boxes:
            rect = mpatches.Rectangle((pred_box.x1, pred_box.y1), pred_box.w, pred_box.h,
                                      linewidth=1, fill=None, edgecolor=PRED_COLORS[pred_box.cls], linestyle='-.')
            ax.add_patch(rect)

    if fig is not None:
        plt.show()


def _pts_to_line_set(pts, connections, color):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(connections)
    line_set.colors = o3d.utility.Vector3dVector([mcolors.to_rgba(color)[:3] for _ in range(len(connections))])
    return line_set


def _create_sphere(x, y, z, r, color):
    sphere = o3d.geometry.create_mesh_sphere(r)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(mcolors.to_rgba(color)[:3])
    sphere.transform(np.array([[1, 0, 0, x],
                               [0, 1, 0, y],
                               [0, 0, 1, z],
                               [0, 0, 0, 1]]))
    return sphere


def view_3D(pts=None, gt_boxes=None, pred_boxes=None, segment_points=False, blocking=True):
    if blocking:
        _view_3D_blocking(pts, gt_boxes, pred_boxes, segment_points)
    else:
        Process(target=_view_3D_blocking, kwargs={'pts': pts, 'gt_boxes': gt_boxes, 'pred_boxes': pred_boxes, 'segment_points': segment_points},
                daemon=True).start()


# segment_points = [False, 'GT', 'PRED']
def _view_3D_blocking(pts=None, gt_boxes=None, pred_boxes=None, segment_points=False):
    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=600)

    # Create coordinate frame at sensor(camera)
    camera_rf = o3d.geometry.create_mesh_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(camera_rf)

    if pts is not None:
        pcd_colors = np.ones(shape=(pts.shape[0], 3)) * 0.85

    # Draw GT Boxes
    if gt_boxes is not None:
        for box in gt_boxes:
            if __is_box_in_workspace(box):
                vis.add_geometry(_pts_to_line_set(pts=box.get_corners(), connections=BOX_CONNECTIONS, color=GT_COLORS[box.cls]))
                vis.add_geometry(_pts_to_line_set(pts=box.get_arrow_pts(), connections=ARROW_CONNECTIONS, color=GT_COLORS[box.cls]))

                if segment_points == 'GT' and pts is not None:
                    pcd_colors[compute_mask_accurate(pts, box)] = mcolors.to_rgba(GT_COLORS[box.cls])[:3]

    # Draw pred_boxes
    if pred_boxes is not None:
        for box in pred_boxes:
            if __is_box_in_workspace(box):
                vis.add_geometry(_pts_to_line_set(pts=box.get_corners(), connections=BOX_CONNECTIONS, color=PRED_COLORS[box.cls]))
                vis.add_geometry(_pts_to_line_set(pts=box.get_arrow_pts(), connections=ARROW_CONNECTIONS, color=PRED_COLORS[box.cls]))

                if segment_points == 'PRED' and pts is not None:
                    pcd_colors[compute_mask_accurate(pts, box)] = mcolors.to_rgba(PRED_COLORS[box.cls])[:3]

    # Create point cloud
    if pts is not None:
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(pts)
        if segment_points is not False:
            pcd.colors = o3d.Vector3dVector(pcd_colors)
        # pcd.colors = o3d.Vector3dVector(np.ones_like(pts) * 0.7)  # Grey pts
        vis.add_geometry(pcd)

    # Change background to black
    vis.get_render_option().background_color = np.array([0, 0, 0])

    # Load camera params (view angle, etc)
    camera_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'open3d_config', 'camera_params.json')
    vis.get_view_control().convert_from_pinhole_camera_parameters(
        o3d.read_pinhole_camera_parameters(camera_params_path))

    # Load render options
    render_options_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'open3d_config', 'render_options.json')
    vis.get_render_option().load_from_json(render_options_path)

    vis.run()
    vis.destroy_window()


def view_BEV(pts=None, gt_boxes=None, pred_boxes=None, scale=1, ax=None, title=''):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
        fig.canvas.set_window_title("Bird's Eye View - " + title)
        fig.set_size_inches(scale * 7, scale * 8)

    # fig.canvas.set_window_title('Bird\'s Eye View')
    ax.set_facecolor((0.9, 0.9, 0.9))
    ax.axis('equal')
    ax.set_xlim(0, 70)
    ax.set_ylim(40, -40)  # Reverse y-axis in plot
    ax.set_xlabel('Z - axis (meters)')
    ax.set_ylabel('X - axis (meters)')

    # Draw points
    if pts is not None:
        ax.scatter(pts[:, 2], pts[:, 0], s=0.9, marker=".", c=[[0.5, 0.5, 0.5]])

    # Draw camera coordinate frame
    ax.add_patch(mpatches.Arrow(0, 0, 3, 0, width=1, facecolor='blue', linewidth=0))
    ax.add_patch(mpatches.Arrow(0, 0, 0, 3, width=1, facecolor='red', linewidth=0))

    # Draw GT Boxes
    if gt_boxes is not None:
        for box in gt_boxes:
            corners = box.get_corners()[0:4, [2, 0]]  # Get x and y coordinates of the 4 corners on the bottom
            ax.add_patch(mpatches.Polygon(corners, fill=False, linewidth=1, edgecolor=GT_COLORS[box.cls], linestyle='-', closed=True))  # Box
            arrow_pts = box.get_arrow_pts()[:, [2, 0]]
            ax.add_patch(mpatches.Polygon(arrow_pts, closed=False, linewidth=1, edgecolor=GT_COLORS[box.cls]))  # Arrow

            if box.text is not None:
                ax.text(arrow_pts[0, 0], arrow_pts[0, 1], box.text, color=GT_COLORS[box.cls], size=10)

    # Draw predicted boxes
    if pred_boxes is not None:
        for box in pred_boxes:
            corners = box.get_corners()[0:4, [2, 0]]  # Get x and y coordinates of the 4 corners on the bottom
            ax.add_patch(mpatches.Polygon(corners, fill=False, linewidth=1, edgecolor=PRED_COLORS[box.cls], linestyle='-', closed=True))  # Box
            arrow_pts = box.get_arrow_pts()[:, [2, 0]]
            ax.add_patch(mpatches.Polygon(arrow_pts, closed=False, linewidth=1, edgecolor=PRED_COLORS[box.cls]))  # Arrow
            ax.add_patch(mpatches.Polygon(arrow_pts, closed=False, linewidth=1, edgecolor=PRED_COLORS[box.cls]))  # Arrow

            if box.text is not None:
                ax.text(arrow_pts[0, 0], arrow_pts[0, 1], box.text, color=PRED_COLORS[box.cls], size=10)

    # if fig is not None:
    #     plt.show()


def view_3D_on_image(img, proj_matrix, gt_boxes=None, pred_boxes=None, scale=1, title='', ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Camera View with 3D Boxes - ' + title)
        fig.set_size_inches(scale * 7, scale * 8)

    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Draw GT boxes in green
    if gt_boxes is not None:
        for gt_box in gt_boxes:
            projected_corners = project(gt_box.get_corners(), proj_matrix)
            for start, end in BOX_CONNECTIONS:
                x1, y1 = projected_corners[start]
                x2, y2 = projected_corners[end]
                ax.add_line(mlines.Line2D([x1, x2], [y1, y2], color=GT_COLORS[gt_box.cls], linestyle='--'))

    # Draw Pred boxes in red
    if pred_boxes is not None:
        for pred_box in pred_boxes:
            projected_corners = project(pred_box.get_corners(), proj_matrix)
            for start, end in BOX_CONNECTIONS:
                x1, y1 = projected_corners[start]
                x2, y2 = projected_corners[end]
                ax.add_line(mlines.Line2D([x1, x2], [y1, y2], color=PRED_COLORS[pred_box.cls], linestyle='--'))

    if fig is not None:
        plt.show()


def render_to_file(fetch_fn, dest_dir):
    render_to_file.added_geometries = []

    def update(_vis):
        # Fetch new frame
        result = fetch_fn()

        # End reached
        if not result:
            print("Done!")
            render_to_file.vis.register_animation_callback(None)
            return False

        print("Rendering ", result['frame_id'])

        # Clear Canvas
        for geometry in render_to_file.added_geometries:
            render_to_file.vis.remove_geometry(geometry)
        render_to_file.added_geometries = []

        # Split result
        gt_boxes = result['gt_boxes']
        pts = result['pts']
        pred_boxes = result['pred_boxes']
        segment_points = 'GT'

        if pts is not None:
            pcd_colors = np.ones(shape=(pts.shape[0], 3)) * 0.85

        # Draw GT Boxes
        if gt_boxes is not None:
            for box in gt_boxes:
                if __is_box_in_workspace(box):
                    render_to_file.added_geometries += [
                        _pts_to_line_set(pts=box.get_corners(), connections=BOX_CONNECTIONS, color=GT_COLORS[box.cls]),
                        _pts_to_line_set(pts=box.get_arrow_pts(), connections=ARROW_CONNECTIONS, color=GT_COLORS[box.cls])]

                    if segment_points == 'GT' and pts is not None:
                        pcd_colors[compute_mask_accurate(pts, box)] = mcolors.to_rgba(GT_COLORS[box.cls])[:3]

        # Draw pred_boxes
        if pred_boxes is not None:
            for box in pred_boxes:
                if __is_box_in_workspace(box):
                    render_to_file.added_geometries += [
                        _pts_to_line_set(pts=box.get_corners(), connections=BOX_CONNECTIONS, color=PRED_COLORS[box.cls]),
                        _pts_to_line_set(pts=box.get_arrow_pts(), connections=ARROW_CONNECTIONS, color=PRED_COLORS[box.cls])]

                    if segment_points == 'PRED' and pts is not None:
                        pcd_colors[compute_mask_accurate(pts, box)] = mcolors.to_rgba(PRED_COLORS[box.cls])[:3]

        # Create point cloud
        if pts is not None:
            pcd = o3d.PointCloud()
            pcd.points = o3d.Vector3dVector(pts)
            if segment_points is not False:
                pcd.colors = o3d.Vector3dVector(pcd_colors)
            render_to_file.added_geometries += [pcd]

        # Add geometries
        for geometry in render_to_file.added_geometries:
            render_to_file.vis.add_geometry(geometry)

        # Change background to black
        render_to_file.vis.get_render_option().background_color = np.array([0, 0, 0])

        # Load camera params (view angle, etc)
        camera_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'open3d_config', 'camera_params_closeup.json')
        render_to_file.vis.get_view_control().convert_from_pinhole_camera_parameters(o3d.read_pinhole_camera_parameters(camera_params_path))

        # Load render options
        render_options_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'open3d_config', 'render_options.json')
        render_to_file.vis.get_render_option().load_from_json(render_options_path)

        # Save to file
        img = render_to_file.vis.capture_screen_float_buffer(True)
        plt.imsave(os.path.join(dest_dir, result['frame_id'] + ".png"), np.asarray(img), dpi=1)

    vis = o3d.visualization.Visualizer()
    render_to_file.vis = vis

    vis.create_window(width=1200, height=600)

    # Create coordinate frame at sensor(camera)
    camera_rf = o3d.geometry.create_mesh_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(camera_rf)

    # Add Callback
    vis.register_animation_callback(update)

    # Run
    vis.run()
    vis.destroy_window()
