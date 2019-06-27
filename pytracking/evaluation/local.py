from pytracking.evaluation.environment import EnvSettings

import os.path as osp
CURRENT_DIR = osp.dirname(__file__)
ROOT_DIR = osp.join(CURRENT_DIR, '..', '..')

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = ''
    settings.lasot_path = ''
    settings.network_path = osp.join(ROOT_DIR, 'pytracking/networks')    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''

    return settings

