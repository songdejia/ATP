#!/usr/bin/python
import os
import numpy as np
import sys

del os.environ['MKL_NUM_THREADS']
from pytracking.evaluation import Tracker
import cv2
import vot
from vot import Polygon, Point

import os.path as osp
CURRENT_DIR = osp.dirname(__file__)
ROOT_DIR = CURRENT_DIR

params = {'tracker_name': 'atp',
          'base_param': 'vot2019',
          'refine': True}

def create_tracker(params):
    tracker_name = params['tracker_name']
    base_param = params['base_param']
    gentracker = Tracker(tracker_name, base_param, 0)
    return gentracker.tracker_class(gentracker.parameters)


def get_axis_aligned_bbox(region):
    # region (1,4,2)
    region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                       region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])

    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])

    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])

    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    x11 = cx - w // 2
    y11 = cy - h // 2

    return x11, y11, w, h


refine = params['refine']
if refine:
    import matlab
    import matlab.engine

    engine = matlab.engine.start_matlab()
    engine.addpath(ROOT_DIR)
else:
    engine = None

handle = vot.VOT("polygon")
gt_bbox = handle.region()
gt_bbox = np.array(gt_bbox).reshape((1, 4, 2))
init_bbox = get_axis_aligned_bbox(gt_bbox)

tracker = create_tracker(params)
image_file = handle.frame()
if not image_file:
    sys.exit(0)
im = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)  # HxWxC
tracker.initialize(im, init_bbox)
tracker.engine = engine

while True:
    image_file = handle.frame()
    if not image_file:
        break
    im = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)  # HxWxC
    pred_bbox = tracker.track(im)
    points = [Point(pred_bbox[0], pred_bbox[1]),
              Point(pred_bbox[2], pred_bbox[3]),
              Point(pred_bbox[4], pred_bbox[5]),
              Point(pred_bbox[6], pred_bbox[7])]
    handle.report(Polygon(points))
