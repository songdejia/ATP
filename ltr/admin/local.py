import os.path as osp
CURRENT_DIR = osp.dirname(__file__)
ROOT_DIR = osp.join(CURRENT_DIR, '..', '..')

class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = osp.join(ROOT_DIR, 'pytracking/networks')    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
