from pytracking.tracker.base import BaseTracker
from pytracking.tracker.atom import ATOM
from pytracking.tracker.atp.utils import *
from pytracking.tracker.siamesemask.siamesemask import SMASK
from pytracking.tracker.siamesemask_127.siamesemask import SEQ_SMASK
from pytracking import dcf
from pytracking.features.preprocessing import numpy_to_torch
import torch
import math

import cv2


class ATP(BaseTracker):
    def initialize(self, image, state, *args, **kwargs):
        image = self.preprocess_image(image)

        # matlab engine
        self.engine = None

        # preprocess params
        self.setting_adaptive_maximal_aspect_ratio(image, state)
        self.setting_adaptive_search_region_using_area_ratio(image, state)
        self.setting_extra_property(image, state)

        # base tracker
        self.atom = ATOM(self.params)
        self.init_atom(image, state)

        # parallel siam mask
        if getattr(self.params, 'use_parallel_smask', False):
            self.smask = SMASK(self.params)
            self.init_parallel_smask(image, state)

        # sequential siam mask
        if getattr(self.params, 'use_sequential_smask', False):
            self.seq_mask = SEQ_SMASK(self.params)
            self.init_sequential_smask(image, state)

    def track(self, image):
        """ state_stage1 is rect/polygon for atom or siammask """
        image = self.preprocess_image(image)

        self.restore_scale_params()

        state_stage2 = state_stage1 = rect_atom = self.track_atom(image)
        assert len(rect_atom) == 4

        if getattr(self.params, 'use_parallel_smask', False):
            polygon_parallel_smask = self.track_para_smask(image)
            state_stage2 = state_stage1 = self.replace_atom_strategy(rect_atom, polygon_parallel_smask, image)

            if getattr(self.params, 'use_smask_replace_atom', False):
                self.replace_atom_strategy_complete(rect_atom, polygon_parallel_smask)
        if self.smask_replace_atom:
            state_stage2 = state_stage1 = polygon_parallel_smask

        assert len(state_stage1) == 8

        if getattr(self.params, 'use_sequential_smask', False):
            polygon_sequential_smask, ratio = self.track_seq_smask(image, state_stage1)
            state_stage2 = self.replace_stage1_strategy(state_stage1, polygon_sequential_smask, ratio)
        assert len(state_stage2) == 8

        if getattr(self.params, 'use_area_ratio_prevent_zoom_in', False):
            self.prevent_zoom_in(image, state_stage2)

        return state_stage2

    def preprocess_image(self, image):
        if getattr(self.params, 'use_histogram_equalization', False):
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            processed_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            processed_img = image
        return processed_img

    def restore_scale_params(self):
        if self.restore > 0:
            self.restore_time -= 1
        if self.restore and self.restore_time == 0:
            self.params.max_image_sample_size = self.restore_scales['max_image_sample_size']
            self.params.min_image_sample_size = self.restore_scales['min_image_sample_size']
            self.params.search_area_scale = self.restore_scales['search_area_scale']
            self.restore = False

    def prevent_zoom_in(self, image, polygon):
        """ use area ratio to prevent zoom in, current_frame_area/last_frame_area < 0.75, enlarge scale """
        blk = np.zeros_like(image)[:, :, 0].astype(np.int32)
        polygon = np.array(polygon).reshape(4, 2).astype(np.int32)
        blk = cv2.fillPoly(blk, [polygon], 1)
        area = np.sum(blk)
        if len(self.seq_smask_areas) > 1:
            if area / self.seq_smask_areas[-1] < self.params.area_ratio_zoom_in_ratio and \
                    self.params.search_area_scale == 4:
                self.params.max_image_sample_size = (22 * 16) ** 2
                self.params.min_image_sample_size = (22 * 16) ** 2
                self.params.search_area_scale = 6
                self.restore = True
                self.restore_time = 2
                self.restore_scales['max_image_sample_size'] = (14 * 16) ** 2
                self.restore_scales['min_image_sample_size'] = (14 * 16) ** 2
                self.restore_scales['search_area_scale'] = 4

        self.seq_smask_areas.append(area)

    def setting_adaptive_maximal_aspect_ratio(self, image, state):
        """
        Input image (Original shape) / state (Rect)

        Return None
        """
        if getattr(self.params, 'use_adaptive_maximal_aspect_ratio', False):
            ratio = max(state[2] / state[3], state[3] / state[2])
            if ratio > self.params.maximal_aspect_ratio:
                self.params.maximal_aspect_ratio = ratio + 1

    def setting_adaptive_search_region_using_area_ratio(self, image, state):
        """
        Input image (Original shape) / state (Rect) 

        Return None
        """
        if getattr(self.params, 'use_area_ratio_adaptive_search_region', False):
            ratio = state[2] * state[3] / (image.shape[0] * image.shape[1])
            if ratio < self.params.area_ratio_adaptive_ratio:
                self.params.max_image_sample_size = (22 * 16) ** 2
                self.params.min_image_sample_size = (22 * 16) ** 2
                self.params.search_area_scale = 6
            else:
                self.params.max_image_sample_size = (14 * 16) ** 2
                self.params.min_image_sample_size = (14 * 16) ** 2
                self.params.search_area_scale = 4

    def setting_adaptive_search_region_using_speed(self, im):
        """ reinitialze search region scale for next frame """
        self.atom.target_scale = 1.0
        search_area = torch.prod(self.atom.target_sz * self.atom.params.search_area_scale).item()

        if search_area > self.atom.params.max_image_sample_size:
            self.atom.target_scale = math.sqrt(search_area / self.atom.params.max_image_sample_size)
        elif search_area < self.atom.params.min_image_sample_size:
            self.atom.target_scale = math.sqrt(search_area / self.atom.params.min_image_sample_size)

        # Target size in base scale
        self.atom.base_target_sz = self.atom.target_sz / self.atom.target_scale

        # Use odd square search area and set sizes
        feat_max_stride = max(self.atom.params.features.stride())
        if getattr(self.atom.params, 'search_area_shape', 'square') == 'square':
            self.atom.img_sample_sz = torch.round(
                torch.sqrt(torch.prod(self.atom.base_target_sz * self.atom.params.search_area_scale))) * torch.ones(2)
        elif self.atom.params.search_area_shape == 'initrect':  # 选的非正方形
            self.atom.img_sample_sz = torch.round(self.atom.base_target_sz * self.atom.params.search_area_scale)
        else:
            raise ValueError('Unknown search area shape')
        if self.atom.params.feature_size_odd:
            self.atom.img_sample_sz += feat_max_stride - self.atom.img_sample_sz % (2 * feat_max_stride)
        else:
            self.atom.img_sample_sz += feat_max_stride - (self.atom.img_sample_sz + feat_max_stride) % (
                        2 * feat_max_stride)

        # Set sizes
        self.atom.img_support_sz = self.atom.img_sample_sz
        self.atom.feature_sz = self.atom.params.features.size(self.atom.img_sample_sz)
        self.atom.output_sz = self.atom.params.score_upsample_factor * self.atom.img_support_sz  # Interpolated size of the output
        self.atom.iou_img_sample_sz = self.atom.img_sample_sz
        # Setup scale bounds
        im = numpy_to_torch(im)
        self.atom.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.atom.min_scale_factor = torch.max(10 / self.atom.base_target_sz)
        self.atom.max_scale_factor = torch.min(self.atom.image_sz / self.atom.base_target_sz)

        self.atom.output_window = None
        if getattr(self.params, 'window_output', False):
            if getattr(self.params, 'use_clipped_window', False):
                self.atom.output_window = dcf.hann2d_clipped(self.atom.output_sz.long(),
                                                             self.atom.output_sz.long() * self.params.effective_search_area / self.params.search_area_scale,
                                                             centered=False).to(self.params.device)
            else:
                self.atom.output_window = dcf.hann2d(self.atom.output_sz.long(), centered=False).to(self.params.device)

    def setting_extra_property(self, image, state):
        # some extra property for added part
        self.center_dist = []  # Max response dist to center in search region for speed
        self.sample_max_response_location = []
        self.tracklet = []  # tracklet for kalman filter
        self.htracklet = []  # tracklet for kalman filter
        self.die = False
        self.image_history = image
        self.rect_final_history = state
        self.count = 0
        self.seq_smask_areas = []
        self.restore = False
        self.restore_time = 0
        self.restore_scales = {}
        self.iou_topk = []  #to judge use smask to replace atom
        self.smask_score_topk = []
        self.smask_replace_atom = False #status to judge use smask to replace atom

    def replace_atom_strategy_complete(self, rect_atom, polygon_parallel_smask):
        if (not self.smask_replace_atom) and len(self.iou_topk) < 10:
            area1, area2, ovr = calculate_area_ovr(rect_2_poly(rect_atom), polygon_parallel_smask)
            self.iou_topk.append(np.around(ovr/(area1+area2-ovr), decimals = 2))
            self.smask_score_topk.append(np.around(self.smask.best_score, decimals = 2))
    
            if np.max(self.iou_topk)>0.7 and np.min(self.iou_topk)<0.3 and \
                np.min(self.smask_score_topk)>=0.9 and len(self.iou_topk) <= 10:
                self.smask_replace_atom = True

    def replace_atom_strategy(self, atom_rect, smask_polygon, image):
        # atom rect mode, smask polygon mode, replace atom in some cases
        # return polygon
        assert len(atom_rect) == 4 and len(smask_polygon) == 8
        x0, y0, w, h = atom_rect
        center = [x0 + w / 2, y0 + h / 2]

        # judge polygon contain atom center 
        h_image, w_image, _ = image.shape
        center_in_mask = point_is_in_poly(center, smask_polygon, h_image, w_image)

        # giou for smask
        atom_polygon = [x0, y0, x0 + w, y0, x0 + w, y0 + h, x0, y0 + h]
        area_atom, area_smask, ovr = calculate_area_ovr(atom_polygon, smask_polygon)

        # First preserve poor smask performance, then replace
        x0, y0, w, h = atom_rect
        final = [x0, y0, x0 + w, y0, x0 + w, y0 + h, x0, y0 + h]
        if area_atom / area_smask <= self.params.parallel_smask_area_preserve_threshold:
            if ovr / area_smask > self.params.parallel_smask_iou_threshold and center_in_mask:
                final = sort_poly(smask_polygon)

        return final

    def replace_stage1_strategy(self, state_stage1, seq_mask_polygon, ratio):
        # state_stage1 rect mode or polygon mode, replace state_stage1 in some cases
        # return rect or polygon(if state_stage1) else polygon(if seq_mask)
        if ratio < self.params.sequential_smask_ratio:
            final = state_stage1
        else:
            final = seq_mask_polygon
        return final

    def rgb2bgr(self, image):
        """atom -> rgb, smask -> gbr """
        return image[:, :, ::-1].copy()

    def init_parallel_smask(self, image, state):
        """ parallel smask should input rect mode region"""
        assert isinstance(state, (list, tuple))
        image = self.rgb2bgr(image)
        if len(state) == 8:
            gt_bbox = np.array(state).reshape((1, 4, 2))
            rect_init_bbox = get_axis_aligned_bbox(gt_bbox)
        else:
            rect_init_bbox = state
        self.smask.initialize(image, rect_init_bbox)

    def init_sequential_smask(self, image, state):
        """ sequential smask should input rect mode region"""
        assert isinstance(state, (list, tuple))
        image = self.rgb2bgr(image)
        if len(state) == 8:
            gt_bbox = np.array(state).reshape((1, 4, 2))
            rect_init_bbox = get_axis_aligned_bbox(gt_bbox)
        else:
            rect_init_bbox = state
        self.seq_mask.initialize(image, rect_init_bbox)

    def init_atom(self, image, state):
        """
        initialize atom 
        image (original)
        state (rect)
        """
        self.atom.initialize(image, state)

    def track_para_smask(self, image):
        image = self.rgb2bgr(image)
        pred_polygon = self.smask.track(image, mask_binary_threshold=0.3)
        return pred_polygon

    def track_seq_smask(self, image, state_stage1):
        image = self.rgb2bgr(image)
        if len(state_stage1) == 8:
            x0, y0, x1, y1, x2, y2, x3, y3 = state_stage1
            if x0 == x3 and x1 == x2 and y0 == y1 and y2 == y3:
                cx, cy = (x0 + x1) / 2, (y0 + y3) / 2
                w, h = x1 - x0, y2 - y0
            else:
                state_stage1 = np.array(state_stage1).reshape((1, 4, 2))
                x0, y0, w, h = get_axis_aligned_bbox(state_stage1)
                cx, cy = x0 + w / 2, y0 + h / 2
        elif len(state_stage1) == 4:
            cx, cy, w, h = rect_2_cxy_wh(state_stage1)
        else:
            raise ValueError

        pred_polygon, ratio = self.seq_mask.track(image, [cx, cy], [w, h], engine=self.engine)

        final = None
        if len(pred_polygon) == 4:
            x0, y0, w, h = pred_polygon
            final = [x0, y0, x0 + w, y0, x0 + w, y0 + h, x0, x0 + h]
        elif len(pred_polygon) == 8:
            final = pred_polygon
        else:
            raise ValueError

        return final, ratio

    def track_atom(self, image):
        # update search region params
        if getattr(self.params, 'use_speed_adaptive_search_region', False):
            self.setting_adaptive_search_region_using_speed(image)

        # track operation
        rect_atom = self.atom.track(image)

        # add use speed adaptive search region
        if getattr(self.params, 'use_speed_adaptive_search_region', False):
            size = self.atom.sample_img.shape[0]
            response = self.atom.response.squeeze().cpu().numpy()
            response = cv2.resize(response, (size, size), interpolation=cv2.INTER_CUBIC)
            self.atom.response = response
            maxindex = np.argmax(response)
            r, c = maxindex // size, maxindex % size
            c_dist = np.sqrt((c - size / 2) ** 2 + (r - size / 2) ** 2) / size
            self.center_dist.append(c_dist)
            self.sample_max_response_location.append([r / size, c / size])

            HistoryLength = len(self.sample_max_response_location)
            speed = mean = 0
            if HistoryLength >= 2:
                r0, c0 = self.sample_max_response_location[-2]
                r1, c1 = self.sample_max_response_location[-1]
                speed = np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)  # current speed
                if HistoryLength > 4:
                    r0, c0 = self.sample_max_response_location[-2]
                    r1, c1 = self.sample_max_response_location[-3]
                    r2, c2 = self.sample_max_response_location[-4]
                    r3, c3 = self.sample_max_response_location[-5]
                    speed1 = np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
                    speed2 = np.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)
                    speed3 = np.sqrt((r3 - r2) ** 2 + (c3 - c2) ** 2)
                    mean = (speed1 + speed2 + speed3) / 3  # mean speed in past 4 frames

            if (speed >= self.params.current_speed_threshold and mean >= self.params.mean_speed_threshold) or (
                    c_dist >= self.params.center_distance_threshold):
                self.atom.params.max_image_sample_size = (30 * 16) ** 2
                self.atom.params.min_image_sample_size = (30 * 16) ** 2
                self.atom.params.search_area_scale = 8
        return rect_atom
