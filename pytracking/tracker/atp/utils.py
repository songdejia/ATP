import numpy as np
import os.path as osp
import cv2
import math


def Normalize(data):
    """
    Input data: Array (m, n)

    return: Normalized Array range from 0 ~ 1 
    """
    assert isinstance(data, np.ndarray), '{} => Normalize => input error'.format(osp.abspath(__file__))
    mn = np.min(data)
    mx = np.max(data - mn)
    nm = (data - mn) / mx
    return nm


def calculate_distance(c1, c2):
    """
    Input c1:[x0, y0]/ c2:[x0, y0]
    return distance between c1 and c2
    """
    assert isinstance(c1, (list, tuple)) and isinstance(c2, (list, tuple)) and len(c1) == 2 and len(c2) == 2, \
        '{} => calculate_distance => input error'.format(osp.abspath(__file__))

    return math.sqrt(math.pow(c1[0] - c2[0], 2) + math.pow(c1[1] - c2[1], 2))


def choose_best_begin_point(coordinate):
    """
    find top-left vertice and resort
    Input coordinate like [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    return: list like [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]
    """
    assert np.array(coordinate).shape == (4, 2), '{} => \
    choose_best_begin_point => input error'.format(osp.abspath(__file__))
    final_result = []

    x1 = coordinate[0][0]
    y1 = coordinate[0][1]
    x2 = coordinate[1][0]
    y2 = coordinate[1][1]
    x3 = coordinate[2][0]
    y3 = coordinate[2][1]
    x4 = coordinate[3][0]
    y4 = coordinate[3][1]
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                 [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
                 [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = calculate_distance(combinate[i][0], dst_coordinate[0]) + calculate_distance(combinate[i][1], \
                                                                                                 dst_coordinate[
                                                                                                     1]) + calculate_distance(
            combinate[i][2], dst_coordinate[2]) + calculate_distance(combinate[i][3], \
                                                                     dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    final_result.append(combinate[force_flag])
    return final_result


def polygon_area(poly):
    '''
    compute area of a polygon
    Input: [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    return:real value
    '''
    assert np.array(poly).shape == (4, 2), '{} => \
    polygon_area => input error'.format(osp.abspath(__file__))
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.


def sort_poly(box):
    '''
    return clockwise box
    Input: [x0, y0, x1, y1, x2, y2, x3, y3]
    return:[x0, y0, x1, y1, x2, y2, x3, y3]
    '''
    assert isinstance(box, list) and len(box) == 8, '{} => \
    sort_poly => input error'.format(osp.abspath(__file__))
    cords = np.array(box).reshape(4, 2)
    poly = choose_best_begin_point(cords)  # 可顺可逆
    poly = np.array(poly).reshape(4, 2)

    p_area = polygon_area(poly)
    if p_area > 0:
        poly = poly[(0, 3, 2, 1), :]

    poly = [poly[0][0], poly[0][1], poly[1][0], poly[1][1], poly[2][0], poly[2][1], poly[3][0], poly[3][1], ]

    return poly


def get_axis_aligned_bbox(region):
    """
    Input region Array(1, 4, 2)
    return Rect x11, y11, w, h
    """
    assert np.array(region).shape == (1, 4, 2), '{} => \
    polygon_area => input error'.format(osp.abspath(__file__))
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


def calculate_area_ovr(box1, box2):
    """
    Input box -- list(x0, y0, x1, y1, x2, y2, x3, y3)
    Return AreaBox1, AreaBox2, AreaOvr
    """
    assert len(box1) == 8 and len(box2) == 8, '{} => \
    calculate_area_ovr => input error'.format(osp.abspath(__file__))
    if isinstance(box1, list):
        box1 = np.array(box1, dtype=np.int32).reshape(4, 2)
    if isinstance(box2, list):
        box2 = np.array(box2, dtype=np.int32).reshape(4, 2)
    # box1:poly shape=[4,2]
    # box2:poly shape=[4,2]

    max_size1 = np.max(np.array(box1))
    max_size2 = np.max(np.array(box2))
    max_size = max(max_size1, max_size2)

    mask1 = np.zeros((max_size, max_size), dtype=np.int32)
    mask2 = np.zeros((max_size, max_size), dtype=np.int32)

    cv2.fillPoly(mask1, [box1], 1)
    cv2.fillPoly(mask2, [box2], 1)

    overlap = np.sum(np.logical_and(mask1, mask2))
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)
    return area1, area2, overlap


def point_is_in_poly(point, poly, h, w):
    """
    Input 
    point (x, y)
    poly  (x0, y0, x1, y1, x2, y2, x3, y3)
    h/w   image shape
    Return whether point in poly
    """
    assert len(point) == 2 and len(poly) == 8, '{} => \
    point_is_in_poly => input error'.format(osp.abspath(__file__))
    if isinstance(poly, list):
        poly = np.array(poly, dtype=np.int32).reshape(4, 2)
    blk = np.zeros((h, w), dtype=np.int32)
    cv2.fillPoly(blk, [poly], 1)
    x, y = int(point[0]), int(point[1])

    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    else:
        return blk[y][x] == 1


def rect_2_cxy_wh(rect):
    """ top-left to cx cy """
    return np.array([rect[0] + rect[2] / 2, rect[1] + rect[3] / 2, rect[2], rect[3]])

def rect_2_poly(rect):
    x0, y0, w, h = rect
    return [x0, y0, x0+w, y0, x0+w, y0+h, x0, y0+h]
