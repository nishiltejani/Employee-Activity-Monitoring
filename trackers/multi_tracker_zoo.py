import os.path
import os
import yaml
from easydict import EasyDict as edict


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
                cfg_dict.update(yaml_)

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            self.update(yaml_)

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)


def create_tracker():
    cfg = get_config()
    cfg.merge_from_file("trackers/bytetrack/configs/bytetrack.yaml")
    from trackers.bytetrack.byte_tracker import BYTETracker
    bytetracker = BYTETracker(
        track_thresh=cfg.bytetrack.track_thresh,
        match_thresh=cfg.bytetrack.match_thresh,
        track_buffer=cfg.bytetrack.track_buffer,
        frame_rate=cfg.bytetrack.frame_rate
    )
    return bytetracker
