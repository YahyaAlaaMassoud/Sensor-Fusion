
import numpy as np
from shapely.geometry import Polygon

def compute_iou_bev(box_1, box_2, axis_aligned):
    if axis_aligned:
        x1_1, x2_1 = box_1.z - box_1.w / 2, box_1.z + box_1.w / 2
        y1_1, y2_1 = box_1.x - box_1.l / 2, box_1.x + box_1.l / 2
        x1_2, x2_2 = box_2.z - box_2.w / 2, box_2.z + box_2.w / 2
        y1_2, y2_2 = box_2.x - box_2.l / 2, box_2.x + box_2.l / 2
        
        area_1 = box_1.w * box_1.l
        area_2 = box_2.w * box_2.l
    
        x1, y1 = max(x1_1, x1_2), max(y1_1, y1_2)
        x2, y2 = min(x2_1, x2_2), min(y2_1, y2_2)
        
        intersection = max(0, (x2 - x1)) * max(0, (y2 - y1))
        iou = intersection / (area_1 + area_2 - intersection)
    else:
        box_1 = Polygon(box_1.get_corners().T[0:4, [2, 0]])
        box_2 = Polygon(box_2.get_corners().T[0:4, [2, 0]])
        iou = box_1.intersection(box_2).area / box_1.union(box_2).area
        
    return iou

def nms_bev(boxes, iou_thresh, max_boxes, min_hit, axis_aligned):
    boxes.sort(key=lambda box: box.confidence, reverse=True)
    boxes = boxes[:max_boxes]

    filtered_boxes = []
    while len(boxes) > 0:
        top_box = boxes[0]
        boxes = np.delete(boxes, 0)  # Remove top box from main list

        # Remove all other boxes overlapping with selected box
        boxes_to_remove = []
        hits = 1
        for box_id in range(len(boxes)):
            iou = compute_iou_bev(boxes[box_id], top_box, axis_aligned)
            if iou > iou_thresh:
                boxes_to_remove += [box_id]
                hits += 1
        boxes = np.delete(boxes, boxes_to_remove)

        # Add box with highest confidence to output list
        if hits >= min_hit:
            filtered_boxes += [top_box]

    return filtered_boxes