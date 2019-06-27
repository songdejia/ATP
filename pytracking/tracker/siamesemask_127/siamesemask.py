import json
import cv2
import torch

from torch.autograd import Variable
from pytracking.tracker.base import BaseTracker
from pytracking.tracker.siamesemask_127.experiments.siammask.custom import Custom
from pytracking.tracker.siamesemask_127.utils.anchors import Anchors
from pytracking.tracker.siamesemask_127.utils.tracker_config import TrackerConfig
from pytracking.tracker.siamesemask_127.utils.bbox_helper import *
import matlab
import math

def rect_2_cxy_wh(rect):
    """ top-left to cx cy """
    return np.array([rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]), np.array([rect[2], rect[3]])  # 0-index


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    # cv2.imshow('crop', im_patch)
    # cv2.waitKey(0)
    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch


def seg_box(engine, anns, gt):
    anns = anns.reshape(-1)
    seg = matlab.double(anns.tolist())
    gt = matlab.double(np.array(gt).tolist())
    bbox = engine.optimize_bboxes(seg, gt)
    bbox = np.array(bbox).reshape((4, 2))
    return bbox


def siamese_track(state, im, engine, vis=False, is_refine=True, is_fast_refine=True, is_faster_refine=True,
                  angle_state=True, soft_angle_state=True, mask_enable=True, refine_enable=True):
    ''' state'''
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']
    ''' state'''

    '''padding'''
    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)

    scale_x = p.exemplar_size / s_x
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    '''padding'''

    '''padding_box  atom_box'''
    crop_box = [target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)]
    temp_atom_box = ([int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2),
                      int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)]).copy()
    temp_crop_box = np.array([int(crop_box[0]), int(crop_box[1]), int(crop_box[0]) + int(crop_box[2]),
                              int(crop_box[1]) + int(crop_box[3])]).copy()
    '''padding_box  atom_box'''

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    if mask_enable:
        score, delta, mask = net.track_mask(x_crop.cuda())
        # print(mask.shape)
    else:
        score, delta = net.track(x_crop.cuda())
    if mask_enable:
        delta_x, delta_y = 4, 4
        if refine_enable:
            mask = net.track_refine((delta_y, delta_x)).cuda().sigmoid().squeeze().view(
                p.out_size, p.out_size).cpu().data.numpy()
        else:
            mask = mask[0, :, delta_y, delta_x].sigmoid(). \
                squeeze().view(p.out_size, p.out_size).cpu().data.numpy()

        # print(mask.shape)
        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c],
                                [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=padding)
            return crop

        s = crop_box[2] / p.instance_size
        sub_box = [crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
                   crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
                   s * p.exemplar_size, s * p.exemplar_size]
        s = p.out_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, state['im_w'] * s, state['im_h'] * s]
        mask_in_img = crop_back(mask, back_box, (state['im_w'], state['im_h']))

        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
        target_mask_iou = target_mask.copy().astype(np.float32)
        target_mask_iou_faster = target_mask.copy().astype(np.float32)

        show_mask = np.stack([target_mask, target_mask, target_mask], -1) * 255
        show_img = im.copy()

        ''' Minimum circumscribed rectangle'''
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(target_pos, target_sz)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        ''' Minimum circumscribed rectangle'''

        '''       refine       '''
        if is_refine:
            try:
                pad_l = np.abs(min(0, temp_crop_box[0]))
                pad_t = np.abs(min(0, temp_crop_box[1]))
                pad_r = max(0, temp_crop_box[2] - target_mask_iou.shape[1])
                pad_b = max(0, temp_crop_box[3] - target_mask_iou.shape[0])
                target_mask_iou2 = np.pad(target_mask_iou, [[pad_t, pad_b], [pad_l, pad_r]], 'constant')
                mat_bbox = rbox_in_img.copy()
                mat_bbox[:, 0] = mat_bbox[:, 0] - temp_crop_box[0]
                mat_bbox[:, 1] = mat_bbox[:, 1] - temp_crop_box[1]
                part_mask = target_mask_iou2[temp_crop_box[1] + pad_t:temp_crop_box[3] + pad_t,
                            temp_crop_box[0] + pad_l:temp_crop_box[2] + pad_l]
                assert part_mask.shape[0] == part_mask.shape[1]
                fin_box = seg_box(engine, part_mask, mat_bbox.reshape(-1))
                fin_box[:, 0] = fin_box[:, 0] + temp_crop_box[0]
                fin_box[:, 1] = fin_box[:, 1] + temp_crop_box[1]
                soft_ratio_mask = 1
            except:
                fin_box = rbox_in_img
                soft_ratio_mask = 1
                print('refine error==========================================================================')
        elif is_fast_refine:
            # start=time.time()
            try:
                angle = cv2.minAreaRect(rbox_in_img.astype(np.int32))[2]
                if -88 < angle < -3:
                    pad_l = np.abs(min(0, temp_crop_box[0]))
                    pad_t = np.abs(min(0, temp_crop_box[1]))
                    pad_r = max(0, temp_crop_box[2] - target_mask_iou.shape[1])
                    pad_b = max(0, temp_crop_box[3] - target_mask_iou.shape[0])
                    target_mask_iou2 = np.pad(target_mask_iou, [[pad_t, pad_b], [pad_l, pad_r]], 'constant')
                    mat_bbox = rbox_in_img.copy()
                    mat_bbox[:, 0] = mat_bbox[:, 0] - temp_crop_box[0]
                    mat_bbox[:, 1] = mat_bbox[:, 1] - temp_crop_box[1]
                    part_mask = target_mask_iou2[temp_crop_box[1] + pad_t:temp_crop_box[3] + pad_t,
                                temp_crop_box[0] + pad_l:temp_crop_box[2] + pad_l]
                    assert part_mask.shape[0] == part_mask.shape[1]
                    base_size = part_mask.shape[0]

                    if base_size <= 64:
                        fin_box = seg_box(engine, part_mask, mat_bbox.reshape(-1))
                        fin_box[:, 0] = fin_box[:, 0] + temp_crop_box[0]
                        fin_box[:, 1] = fin_box[:, 1] + temp_crop_box[1]

                    elif base_size <= 128:
                        ratio_small = base_size / 64.
                        part_mask_small = cv2.resize(part_mask, (64, 64))
                        mat_bbox_small = mat_bbox / ratio_small
                        fin_box_small = seg_box(engine, part_mask_small, mat_bbox_small.reshape(-1))
                        fin_box = fin_box_small * ratio_small
                        fin_box[:, 0] = fin_box[:, 0] + temp_crop_box[0]
                        fin_box[:, 1] = fin_box[:, 1] + temp_crop_box[1]

                    elif base_size <= 256:
                        ratio_small = 2
                        new_s = int(base_size / 2)
                        part_mask_small = cv2.resize(part_mask, (new_s, new_s))
                        mat_bbox_small = mat_bbox / ratio_small
                        fin_box_small = seg_box(engine, part_mask_small, mat_bbox_small.reshape(-1))
                        fin_box = fin_box_small * ratio_small
                        fin_box[:, 0] = fin_box[:, 0] + temp_crop_box[0]
                        fin_box[:, 1] = fin_box[:, 1] + temp_crop_box[1]

                    else:
                        fin_box = rbox_in_img
                    soft_ratio_mask = 1
                else:
                    fin_box = rbox_in_img
                    soft_ratio_mask = -1
            except:
                fin_box = rbox_in_img
                soft_ratio_mask = 1
                print('refine error==========================================================================')
            # print('time={}'.format(time.time()-start))
        elif is_faster_refine:
            try:
                angle = cv2.minAreaRect(rbox_in_img.astype(np.int32))[2]
                if -88 < angle < -3:
                    x_max = int(np.max(rbox_in_img[:, 0])) + 1
                    y_max = int(np.max(rbox_in_img[:, 1])) + 1

                    x_min = int(np.min(rbox_in_img[:, 0]))
                    y_min = int(np.min(rbox_in_img[:, 1]))

                    w = x_max - x_min
                    h = y_max - y_min
                    s = max(w, h)
                    cx = (x_max + x_min) / 2
                    cy = (y_max + y_min) / 2
                    temp_crop_box_samll = [math.ceil(cx - s / 2)-2, math.ceil(cy - s / 2)-2, math.ceil(cx + s / 2) + 1, math.ceil(cy + s / 2) + 1]

                    pad_l = np.abs(min(0, temp_crop_box_samll[0]))
                    pad_t = np.abs(min(0, temp_crop_box_samll[1]))
                    pad_r = max(0, temp_crop_box_samll[2] - target_mask_iou_faster.shape[1])
                    pad_b = max(0, temp_crop_box_samll[3] - target_mask_iou_faster.shape[0])
                    target_mask_iou2 = np.pad(target_mask_iou_faster, [[pad_t, pad_b], [pad_l, pad_r]], 'constant')
                    mat_bbox = rbox_in_img.copy()
                    mat_bbox[:, 0] = mat_bbox[:, 0] - temp_crop_box_samll[0]
                    mat_bbox[:, 1] = mat_bbox[:, 1] - temp_crop_box_samll[1]
                    part_mask = target_mask_iou2[temp_crop_box_samll[1] + pad_t:temp_crop_box_samll[3] + pad_t,
                                temp_crop_box_samll[0] + pad_l:temp_crop_box_samll[2] + pad_l]
                    assert part_mask.shape[0] == part_mask.shape[1]
                    base_size = part_mask.shape[0]

                    if base_size <= 32:
                        fin_box = seg_box(engine, part_mask, mat_bbox.reshape(-1))
                        fin_box[:, 0] = fin_box[:, 0] + temp_crop_box_samll[0]
                        fin_box[:, 1] = fin_box[:, 1] + temp_crop_box_samll[1]

                    elif base_size <= 200:
                        ratio_small = 2
                        new_s = int(base_size / 2)
                        part_mask = cv2.resize(part_mask, (new_s, new_s))
                        mat_bbox_small = mat_bbox / ratio_small
                        fin_box_small = seg_box(engine, part_mask, mat_bbox_small.reshape(-1))
                        fin_box = fin_box_small * ratio_small

                        fin_box[:, 0] = fin_box[:, 0] + temp_crop_box_samll[0]
                        fin_box[:, 1] = fin_box[:, 1] + temp_crop_box_samll[1]

                    else:
                        fin_box = rbox_in_img
                    soft_ratio_mask = 1
                else:
                    fin_box = rbox_in_img
                    soft_ratio_mask = -1
            except:
                fin_box = rbox_in_img
                soft_ratio_mask = 1
                print('refine error==========================================================================')
                raise
        else:
            fin_box = rbox_in_img
            soft_ratio_mask = 1
        '''       refine       '''
        try:
            '''limit little mask'''
            angle = cv2.minAreaRect(fin_box.astype(np.int32))[2]
            temp_mask = np.zeros_like(target_mask_iou)
            cv2.fillPoly(temp_mask, [fin_box.astype(np.int32)], 1)
            target_mask_iou = np.logical_and(target_mask_iou, temp_mask).astype(np.float32)
            limit_mask = (
            target_mask_iou[temp_atom_box[1]:temp_atom_box[3], temp_atom_box[0]:temp_atom_box[2]]).copy().astype(
                np.float32)
            mask_area = np.sum(limit_mask)
            rect_area = limit_mask.shape[0] * limit_mask.shape[1]
            ratio_mask = mask_area / rect_area
            '''limit little mask'''

            '''limit angle'''
            if soft_angle_state:
                if soft_ratio_mask == -1:
                    ratio_mask = soft_ratio_mask
            if angle_state:
                if angle < -85 or angle > -5:
                    ratio_mask = -1
            '''limit angle'''

            ''' visual'''
            if vis:
                img_mask_crop = np.maximum(show_img, show_mask)
                img_mask_crop = cv2.addWeighted(show_img, 0.5, img_mask_crop, 0.5, 0)
                # cv2.polylines(img_mask_crop,[gt_box.astype(np.int32)],True,(0,0,255),2)
                cv2.rectangle(img_mask_crop, (temp_atom_box[0], temp_atom_box[1]), (temp_atom_box[2], temp_atom_box[3]),
                              (0, 255, 0), 1)
                cv2.polylines(img_mask_crop, [rbox_in_img.astype(np.int32)], True, (0, 255, 255), 1)
                cv2.polylines(img_mask_crop, [fin_box.astype(np.int32)], True, (255, 255, 0), 1)
                img_mask_crop = cv2.resize(
                    img_mask_crop[temp_crop_box[1]:temp_crop_box[3], temp_crop_box[0]:temp_crop_box[2], :], (127, 127))
            else:
                img_mask_crop = show_mask
            ''' visual'''
        except:
            ratio_mask = 0
            img_mask_crop = show_mask

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    state['target_pos'] = target_pos  # cx
    state['target_sz'] = target_sz
    state['score'] = score
    state['mask'] = mask_in_img if mask_enable else []
    state['ploygon'] = fin_box if mask_enable else []
    return state, ratio_mask


def generate_anchor(cfg, score_size):
    anchors = Anchors(cfg)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)

    total_stride = anchors.stride
    anchor_num = anchor.shape[0]

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def siamese_init(im, target_pos, target_sz, model, hp=None):
    state = dict()
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    p = TrackerConfig()
    p.update(hp, model.anchors)

    p.renew()

    net = model
    p.scales = model.anchors['scales']
    p.ratios = model.anchors['ratios']
    p.anchor_num = len(p.ratios) * len(p.scales)
    p.anchor = generate_anchor(model.anchors, p.score_size)

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    net.template(z.cuda())

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state


class SEQ_SMASK(BaseTracker):
    def initialize(self, image, state, *args, **kwargs):
        # self.config = self.params.config
        # self.ckpt   = self.params.ckpt
        self.sequential_smask_config = self.params.sequential_smask_config
        self.sequential_smask_ckpt = self.params.sequential_smask_ckpt
        self.is_refine = self.params.is_refine
        self.is_fast_refine = self.params.is_fast_refine
        self.is_faster_refine = self.params.is_faster_refine
        self.angle_state = self.params.angle_state
        self.soft_angle_state = self.params.soft_angle_state
        self.cfg = json.load(open(self.sequential_smask_config))
        self.mask_enable = True
        self.refine_enable = True

        model = Custom(anchors=self.cfg['anchors'])
        model.eval()
        model = model.cuda()
        model.load_state_dict(torch.load(self.sequential_smask_ckpt))

        # bbox初始化
        rect_init_gt = state
        target_pos, target_sz = rect_2_cxy_wh(rect_init_gt)  # cx, cy, w, h

        # 初始化
        self.state = siamese_init(image, target_pos, target_sz, model, self.cfg['hp'])

        self.location = cxy_wh_2_rect(self.state['target_pos'], self.state['target_sz'])

    def track(self, image, pos_ex, sz_ex, engine, vis=False):
        self.state['target_pos'] = pos_ex
        self.state['target_sz'] = sz_ex
        self.state, ratio_mask = siamese_track(self.state, image, engine, vis, self.is_refine, self.is_fast_refine,
                                               self.is_faster_refine, self.angle_state, self.soft_angle_state,
                                               self.mask_enable, self.refine_enable)  # track
        if self.mask_enable:
            location = self.state['ploygon'].flatten()
            mask = self.state['mask']
        else:
            location = cxy_wh_2_rect(self.state['target_pos'], self.state['target_sz'])
            mask = []

        if self.mask_enable:
            pred_polygon = [location[0], location[1], location[2], location[3],
                            location[4], location[5], location[6], location[7]]
        else:
            pred_polygon = ((location[0], location[1]),
                            (location[0] + location[2], location[1]),
                            (location[0] + location[2], location[1] + location[3]),
                            (location[0], location[1] + location[3]))

        return pred_polygon, ratio_mask
