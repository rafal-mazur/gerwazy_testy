
# in case of any problems with decoding https://stackoverflow.com/questions/55583306/decoding-geometry-output-of-east-text-detection

import cv2
import depthai as dai
import numpy as np

_conf_threshold: float = 0.5

def to_tensor_result(packet: dai.NNData) -> dict:
    return {tensor.name: np.array(packet.getLayerFp16(tensor.name)) for tensor in packet.getRaw().tensors}


def get_cv_rotated_rect(bbox: tuple[float, float, float, float], angle: float) -> tuple[list[list, list], list[list, list], float]:
    x0, y0, x1, y1 = bbox
    width = abs(x0 - x1)
    height = abs(y0 - y1)
    x = x0 + width * 0.5
    y = y0 + height * 0.5
    return ([x.tolist(), y.tolist()], [width.tolist(), height.tolist()], np.rad2deg(angle))


def rotated_Rectangle(bbox: tuple[float], angle: float) -> np.array:
    X0, Y0, X1, Y1 = bbox
    width = abs(X0 - X1)
    height = abs(Y0 - Y1)
    x = int(X0 + width * 0.5)
    y = int(Y0 + height * 0.5)
    
    tmpw: float = width / 2
    tmph: float = height / 2
    
    pt1_1 = (int(x + tmpw), int(y + tmph))
    pt2_1 = (int(x + tmpw), int(y - tmph))
    pt3_1 = (int(x - tmpw), int(y - tmph))
    pt4_1 = (int(x - tmpw), int(y + tmph))

    del tmpw
    del tmph

    cos = np.cos(angle)
    sin = np.sin(angle)
    
    t = np.array([[cos, -sin, x - x * cos + y * sin],
                  [sin, cos, y - x * sin - y * cos],
                  [0, 0, 1]])

    tmp_pt1_1 = np.array([[pt1_1[0]], [pt1_1[1]], [1]])
    tmp_pt1_2 = np.dot(t, tmp_pt1_1)
    pt1_2 = (int(tmp_pt1_2[0][0]), int(tmp_pt1_2[1][0]))

    tmp_pt2_1 = np.array([[pt2_1[0]], [pt2_1[1]], [1]])
    tmp_pt2_2 = np.dot(t, tmp_pt2_1)
    pt2_2 = (int(tmp_pt2_2[0][0]), int(tmp_pt2_2[1][0]))

    tmp_pt3_1 = np.array([[pt3_1[0]], [pt3_1[1]], [1]])
    tmp_pt3_2 = np.dot(t, tmp_pt3_1)
    pt3_2 = (int(tmp_pt3_2[0][0]), int(tmp_pt3_2[1][0]))

    tmp_pt4_1 = np.array([[pt4_1[0]], [pt4_1[1]], [1]])
    tmp_pt4_2 = np.dot(t, tmp_pt4_1)
    pt4_2 = (int(tmp_pt4_2[0][0]), int(tmp_pt4_2[1][0]))

    points = np.array([pt1_2, pt2_2, pt3_2, pt4_2])

    return points


def non_max_suppression(boxes, probs=None, angles=None, overlapThresh=0.3) :
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int"), angles[pick]


def decode_predictions(scores: np.ndarray, 
                       bboxes_data: np.ndarray, 
                       angles_data: np.ndarray) -> tuple[list[tuple[float, float, float, float]], list[float], list[float]]:
    # Bounding box in EAST
    # 
    #   -------------------------------------------
    #   |                       |                 |
    #   |          height_top   |                 |
    #   |                       |                 |
    #   |                       |                 |
    #   |                       |                 |
    #   |     width_left        |  width_right    |
    #   |-----------------------O-----------------|
    #   |                    offset               |
    #   |                       |                 |
    #   |                       |                 |
    #   |                       |                 |
    #   |           height_low  |                 |
    #   |                       |                 |
    #   |                       |                 |
    #   -------------------------------------------
    #
    # + apply rotation of given angle counterclockwise
    #
    # offset is the point around which rectangle rotates
    
    # scores.shape =      (1, 1, 64, 64)
    # bboxes_data.shape = (1, 4, 64, 64)
    # angles_data.shape = (1, 1, 64, 64)
   
    
    # grab the number of rows and columns from the scores volume, then
    # initialize set of bounding box rectangles, corresponding
    # confidence scores and angles
    (numRows, numCols) = scores.shape[2:4]
    rects: list = []
    confidences: list = []
    angles: list = []

    # loop over the number of rows
    for y in range(0, numRows):
        scoresData: np.ndarray      = scores[0, 0, y]         # scores each box got
        heights_top: np.ndarray     = bboxes_data[0, 0, y]    # list of distances from offset to upper side
        widths_right: np.ndarray    = bboxes_data[0, 1, y]    # list of distances from offset to right side
        heights_low: np.ndarray     = bboxes_data[0, 2, y]    # list of distances from offset to lower side
        widths_left: np.ndarray     = bboxes_data[0, 3, y]    # list of distances from offset to left side
        angles_sample: np.ndarray   = angles_data[0, 0, y]    # angles
        
        # loop over the number of columns
        for x in range(0, numCols):
            if scoresData[x] < _conf_threshold:
                continue

            # input image is 256x256 while feature map is 64x64 so we multiply by 4.0 to get to the right size
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle: float = angles_sample[x]
            cos: float = np.cos(angle)
            sin: float = np.sin(angle)
            
            h: float = heights_top[x] + heights_low[x] # height
            w: float = widths_right[x] + widths_left[x] # width

            # compute both the starting and ending (x, y)-coordinates after rotation
            # for the text prediction bounding box
            bottomrightx = int(offsetX + (cos * widths_right[x]) + (sin * heights_low[x]))
            bottomlefty = int(offsetY - (sin * widths_right[x]) + (cos * heights_low[x])) 
            topleftx = int(bottomrightx - w) # 
            toplefty = int(bottomlefty - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((topleftx, toplefty, bottomrightx, bottomlefty))
            confidences.append(scoresData[x])
            angles.append(angle)

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences, angles)


def decode_east(NN_output: dai.NNData) -> tuple[list, list]:
    scores = NN_output.get_tensor(0)
    bboxes_data = NN_output.get_tensor(1)
    angles_data = NN_output.get_tensor(2)
    bboxes, confs, angles = decode_predictions(scores, bboxes_data, angles_data)
    boxes, angles = non_max_suppression(np.array(bboxes), probs=confs, angles=np.array(angles))
    boxesangles = (boxes, angles)
    return boxesangles


def draw_bboxes(boxesangles: tuple[list, list], frame: np.ndarray) -> np.ndarray:
    bboxes = boxesangles[0]
    angles = boxesangles[1]
    for ((X0, Y0, X1, Y1), angle) in zip(bboxes, angles):
        width = abs(X0 - X1)
        height = abs(Y0 - Y1)
        cX = int(X0 + width * 0.5)
        cY = int(Y0 + height * 0.5)

        rotRect = ((cX, cY), ((X1 - X0), (Y1 - Y0)), angle * (-1))
        points = rotated_Rectangle(frame, rotRect, color=(255, 0, 0), thickness=1)
        cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_8)

    return frame


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
