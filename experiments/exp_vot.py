import sys
sys.path.append('/home/makalo/workspace/songdejia/workspace/CVPR/baidu/vis-var/pytorch-atom/')
import csv
import functools
import os
import os.path as osp
import errno
import numpy as np
import cv2 as cv
import time
from pysot.toolkit.datasets import DatasetFactory
from pysot.toolkit.utils.region import vot_overlap
from pytracking.evaluation import Tracker
from tqdm import tqdm

import multiprocessing
from multiprocessing import Pool
import warnings
import cv2

warnings.filterwarnings("ignore", category=DeprecationWarning)
DATA_ROOT = 'datasets/vot2019'
RESULT_ROOT = 'tracking_results/'
REPEAT  = 15
NUM_GPU = 1
NUM_PROCESS = 1
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

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:    # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def read_image(x):
    return cv.cvtColor(x, cv.COLOR_BGR2RGB)

def create_tracker(params):
    tracker_name = params['tracker_name']
    base_param = params['base_param']
    gentracker = Tracker(tracker_name, base_param, 0)
    return gentracker.tracker_class(gentracker.parameters)

# VOT18
def run_one_sequence(save_dir, params, video):
    #if video.name not  in ['book']:
    #    return
    ResetCount = 0
    idt = multiprocessing.current_process()._identity[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(idt % NUM_GPU)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    save_sub_dir = osp.join(save_dir, 'baseline', video.name)
    mkdir_p(save_sub_dir)
    refine = False
    """
    if refine:
        import matlab
        import matlab.engine
        engine = matlab.engine.start_matlab()
    else:
        engine = None
    """
    engine = None
    for repeat_idx in range(1, REPEAT + 1):
        # visualization
        save_img_path='./tracking_results/'+params['tracker_name']+'/'+params['base_param']+'_vis/'+video.name+'/'+str(repeat_idx)

        # track record
        save_path = osp.join(save_sub_dir, video.name + '_{:03d}.txt'.format(repeat_idx))
        if osp.exists(save_path):
            print('Have Done {}'.format(save_path))
            continue

        frame_counter = 0
        pred_bboxes = []
        for idx, (img_p, gt_bbox) in enumerate(video):
            print('Frame', idx)
            if idx == frame_counter:
                image = read_image(img_p)
                tracker = create_tracker(params)
                if len(gt_bbox) == 8:
                    gt_bbox = np.array(gt_bbox).reshape((1, 4, 2))
                    init_bbox = get_axis_aligned_bbox(gt_bbox)
                elif len(gt_bbox) == 4:
                    gt_x0, gt_y0, gt_w, gt_h = gt_bbox
                    gt_polygon = [gt_x0, gt_y0, gt_x0+gt_w, gt_y0, gt_x0+gt_w, gt_y0+gt_h, gt_x0, gt_y0+gt_h]
                    init_bbox = gt_bbox
                else:
                    raise NotImplementedError
                tracker.initialize(image, init_bbox)
                tracker.engine=engine
                pred_bboxes.append(1)

            elif idx > frame_counter:
                # get tracking result here
                image = read_image(img_p)
                pred_bbox= tracker.track(image)

                if isinstance(pred_bbox, list):
                    pass
                else:
                    pred_bbox = [pred_bbox[0][0],pred_bbox[0][1],pred_bbox[1][0],pred_bbox[1][1],pred_bbox[2][0],pred_bbox[2][1],pred_bbox[3][0],pred_bbox[3][1]]

                overlap = vot_overlap(pred_bbox, gt_bbox, (image.shape[1], image.shape[0]))#输入需要分别为4，8的列表
                debug_info = 'Tracker:{} Video:{} RepeatIDx:{} Frame:{} \nPred:{}\nGt:{}\nOvr:{:.2f}\n'.format(params['base_param'], video.name,repeat_idx,idx,pred_bbox,gt_bbox,overlap)
                #if idx % 10 == 0:
                #    print(debug_info)
                if overlap > 0:
                    # continue tracking
                    pred_bboxes.append(pred_bbox)
                else:
                    if video.name not in ['agility', 'hand2', 'dribble']:
                        print('Fail Restart @ Video:{} Repeat:{} Frame:{:04d} @ scale:{}'.format(video.name, repeat_idx+1, idx, tracker.params.search_area_scale))
                    with open('DebugCrash_{}.txt'.format(params['base_param']), 'a') as f:
                        f.write('Video:{} Frame:{:04d} @ Fail @ scale:{}\n'.format(video.name, idx, tracker.params.search_area_scale))
                    ResetCount += 1
                    del tracker
                    pred_bboxes.append(2)
                    frame_counter = idx + 5
            else:
                pred_bboxes.append(0)

        with open(save_path, 'w') as f:
            outputs = []
            for res in pred_bboxes:
                if isinstance(res, int):
                    outputs.append('{}'.format(res))
                else:
                    if len(res) == 4:
                        outputs.append('{},{},{},{}'.format(res[0], res[1], res[2], res[3]))
                    elif len(res) == 8:
                        outputs.append('{},{},{},{},{},{},{},{}'.format(res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]))
                    else:
                        raise ValueError
            f.write('\n'.join(outputs))

def run_tracker(dataset, params, save_dir):
    with Pool(processes=NUM_PROCESS) as pool:
        for ret in tqdm(pool.imap_unordered(
                functools.partial(run_one_sequence, save_dir, params),
                list(dataset.videos.values()))):
            pass


def main():
    params = {'tracker_name': 'atompp',
            'base_param': 'vot2019_t8_b10'}

    dataset = DatasetFactory.create_dataset(name='VOT2019', dataset_root=DATA_ROOT, load_img=False)
    ct = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

    save_dir = osp.join(RESULT_ROOT, params['tracker_name'], params['base_param'])
    run_tracker(dataset, params, save_dir)

if __name__ == '__main__':
    main()




