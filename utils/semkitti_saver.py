import yaml
import numpy as np
from utils.saver import Saver


class SemkittiSaver(Saver):
    def __init__(self, args):
        super().__init__(args)
        self.coords_dirname = 'velodyne'
        self.ext = 'bin'
        self.data_list, self.original_selections \
            = self.load_data_list()
        with open("dataloader/semantic_kitti/semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']

        self.LABEL_TO_COLOR = np.asarray(
            [
                [100, 150, 245],  # 'car'
                [100, 230, 245],  # 'bicycle'
                [30, 60, 150],  # 'motorcycle'
                [80, 30, 180],  # 'truck'
                [0, 0, 255],  # 'other-vehicle'
                [255, 30, 30],  # 'person'
                [255, 40, 200],  # 'bicyclist'
                [150, 30, 90],  # 'motorcyclist'
                [255, 0, 255],  # 'road'
                [255, 150, 255],  # 'parking'
                [75, 0, 75],  # 'sidewalk'
                [175, 0, 75],  # 'other-ground'
                [255, 200, 0],  # 'building'
                [255, 120, 50],  # 'fence'
                [0, 175, 0],  # 'vegetation'
                [135, 60, 0],  # 'trunk'
                [150, 240, 80],  # 'terrain'
                [255, 240, 150],  # 'pole'
                [255, 0, 0],  # 'traffic-sign'
                [128, 128, 128],  # unlabelled .->. gray
            ]
        )
        self.ignore_label = 19
        print("init done.")

    def load_data(self, fn):
        coords_fn = fn
        labels_fn = fn.replace('velodyne', 'labels').replace('bin', 'label')
        supvox_fn = fn.replace('velodyne', 'supervoxel')
        coords = np.fromfile(coords_fn, dtype=np.float32).reshape(-1, 4)[:, :3]
        rgb = np.ones_like(coords).astype(np.uint8) * 128
        labels = np.fromfile(labels_fn, dtype=np.int32).reshape(-1)
        labels = labels & 0xFFFF
        labels = np.vectorize(self.learning_map.__getitem__)(labels).astype(np.uint8)
        supvox = np.fromfile(supvox_fn, dtype=np.int32).reshape(-1)
        return coords, rgb, labels, supvox
