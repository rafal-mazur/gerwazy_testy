"""Module with functons that help handle EAST network"""
# in case of any problems with decoding https://stackoverflow.com/questions/55583306/decoding-geometry-output-of-east-text-detection

import cv2
import depthai as dai
import numpy as np
from geometry import Point, RotatedRect

_conf_thresh: float = 0.5
_overlap_thresh: float = 0.8


def decode_east(east_output: dai.NNData):
    """Decode output of an EAST detection network to operable data

    Parameters
    ----------
    ``east_output`` : dai.NNData

    Returns
    -------
    list[tuple[RotatedRect, float]]
        List od pairs consisting of bouding box and its score
    """
    # scores.shape =      (1, 1, 64, 64)
    # bboxes_data.shape = (1, 4, 64, 64)
    # angles_data.shape = (1, 1, 64, 64)
   
    east_output_tensor = to_tensor_result(east_output)
    
    scores: np.array = np.reshape(east_output_tensor['feature_fusion/Conv_7/Sigmoid'][0, 0], (64, 64)) # confidences
    bboxes_data: np.array = np.reshape(east_output_tensor['feature_fusion/mul_6'][0], (4, 64, 64)) # rectangular bounding boxes
    angles_data: np.array = np.reshape(east_output_tensor['feature_fusion/sub/Fused_Add_'][0, 0], (64, 64)) # angles to rotate boxes by
    
    # dimensions of a feature map
    (numRows, numCols) = scores.shape
    bboxes: list = []
    confidences: list = []
    angles: list = []

    # loop over the number of rows
    for y in range(0, numRows):
        scores_row: np.ndarray      = scores[y]         # scores each box got
        heights_top: np.ndarray     = bboxes_data[0, y] # list of distances from offset to upper side
        widths_right: np.ndarray    = bboxes_data[1, y] # list of distances from offset to right side
        heights_low: np.ndarray     = bboxes_data[2, y] # list of distances from offset to lower side
        widths_left: np.ndarray     = bboxes_data[3, y] # list of distances from offset to left side
        angles_sample: np.ndarray   = angles_data[y]    # angles
        
        # loop over the number of columns
        for x in range(numCols):
            if scores_row[x] < _conf_thresh:
                continue

            # input image is 256x256 while feature map is 64x64 so we multiply by 4.0 to get to the right size
            offsetX, offsetY = x * 4.0, y * 4.0

            angle: float = angles_sample[x]
            cos: float = np.cos(angle)
            sin: float = np.sin(angle)
            
            h: float = heights_top[x] + heights_low[x] # height
            w: float = widths_right[x] + widths_left[x] # width

            # compute both the starting and ending (x, y)-coordinates after rotation
            # for the text prediction bounding box
            bottomleftx = offsetX + (cos * widths_right[x]) + (sin * heights_low[x])
            bottomlefty = offsetY - (sin * widths_right[x]) + (cos * heights_low[x])

            # add the bounding box coordinates, angles and probability score
            # to our respective lists
            bboxes.append((bottomleftx - w, bottomlefty - h, bottomleftx, bottomlefty))
            angles.append(angle)
            confidences.append(scores_row[x])
        
        # apply non max supression and yield the results
        for bbox, angle, conf in non_max_suppression(np.ndarray(bboxes), np.ndarray(confidences), np.ndarray(angles)):
            yield (RotatedRect(Point(bbox[0], bbox[1]), Point(bbox[2], bbox[3]), angle), conf)



def to_tensor_result(packet: dai.NNData) -> dict:
    return {tensor.name: np.array(packet.getLayerFp16(tensor.name)) for tensor in packet.getRaw().tensors}


def non_max_suppression(boxes: np.ndarray[float, float, float, float], 
                        scores: np.ndarray[float], 
                        angles: np.ndarray[float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]:
    """Apply non max supression to given data

    Parameters
    ----------
    ``boxes`` : np.ndarray[float, float, float, float]
        (x0, y0, x1, y1) coordinates of bounding box
    ``scores`` : np.ndarray[float]
        Confidences of each prediction
    ``angles`` : np.ndarray[float] | None, optional
        Rotation angles, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]
        Filtered ``(boxes, scores, angles)`` if ``angles`` is not None, ``(boxes, scores)`` otherwise 
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    picked: list = []

    # grab the coordinates of the bounding boxes
    x1: np.ndarray = boxes[:, 0]
    y1: np.ndarray = boxes[:, 1]
    x2: np.ndarray = boxes[:, 2]
    y2: np.ndarray = boxes[:, 3]

    # compute the areas of the bounding boxes
    areas = (x2 - x1) * (y2 - y1)
    
    # sort the indexes
    order = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes list
    while len(order) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        i = order[-1]
        order_but_last = order[:-1]
        picked.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[order_but_last])
        yy1 = np.maximum(y1[i], y1[order_but_last])
        xx2 = np.minimum(x2[i], x2[order_but_last])
        yy2 = np.minimum(y2[i], y2[order_but_last])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap
        overlap = (w * h) / areas[order_but_last]

        # delete all indexes from the index list that have overlap greater than the provided overlap threshold
        order = np.delete(order, np.concatenate(([len(order_but_last)], np.where(overlap > _overlap_thresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[picked], scores[picked] if angles is None else boxes[picked], scores[picked], angles[picked]


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
