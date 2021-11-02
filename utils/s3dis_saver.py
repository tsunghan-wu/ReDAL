import numpy as np
from utils.saver import Saver


class S3DISSaver(Saver):
    def __init__(self, args):
        super().__init__(args)
        self.coords_dirname = 'coords'
        self.ext = 'npy'
        self.data_list, self.original_selections \
            = self.load_data_list()

        self.LABEL_TO_COLOR = np.asarray(
            [
                [0, 255, 0],  # 'ceiling'
                [0, 0, 255],  # 'floor'
                [0, 255, 255],  # 'wall'
                [255, 255, 0],  # 'beam'
                [255, 0, 255],  # 'column'
                [100, 100, 255],  # 'window'
                [200, 200, 100],  # 'door'
                [255, 0, 0],  # 'chair'
                [170, 120, 200],  # 'table'
                [10, 200, 100],  # 'bookcase'
                [200, 100, 100],  # 'sofa'
                [200, 200, 200],  # 'board'
                [50, 50, 50],  # 'clutter'
                [128, 128, 128],  # unlabelled .->. gray
            ]
        )
        self.ignore_label = 13
        print("init done.")

    def load_data(self, fn):
        coords_fn = fn
        labels_fn = fn.replace('coords', 'labels')
        rgb_fn = fn.replace('coords', 'rgb')
        supvox_fn = fn.replace('coords', 'supervoxel')
        coords = np.load(coords_fn).astype(np.float32)
        rgb = np.load(rgb_fn).astype(np.float32).astype(np.uint8)
        labels = np.load(labels_fn).astype(np.int32).reshape(-1)
        supvox = np.load(supvox_fn).astype(np.int32).reshape(-1)
        return coords, rgb, labels, supvox
