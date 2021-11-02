import os
import json
import numpy as np
from plyfile import PlyData, PlyElement


root = "/tmp3/tsunghan/S3DIS_processed"
visualize_root = "visualize_init"
fname = "../dataloader/s3dis/init_data/init_label_scan.json"
os.makedirs(visualize_root, exist_ok=True)
# Read List
with open(fname, 'r') as f:
    data_list = json.load(f)


def load_data(fn):
    coords_fn = fn
    feats_fn = fn.replace('coords', 'rgb')
    labels_fn = fn.replace('coords', 'labels')
    supvox_fn = fn.replace('coords', 'supervoxel')
    coords = np.load(coords_fn).astype(np.float32)
    feats = np.load(feats_fn).astype(np.float32)
    labels = np.load(labels_fn).astype(np.int32).reshape(-1)
    supvox = np.load(supvox_fn).astype(np.int32).reshape(-1)
    return coords, feats, labels, supvox


def to_ply(pos, colors, fname):
    # to ply
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    el = PlyElement.describe(ply_array, 'vertex')
    PlyData([el], text=False).write(fname)


OBJECT_COLOR = np.asarray(
    [
        [0, 255, 0],  # 'ceiling' .-> .yellow
        [0, 0, 255],  # 'floor' .-> . blue
        [0, 255, 255],  # 'wall'  ->  brown
        [255, 255, 0],  # 'beam'  ->  salmon
        [255, 0, 255],  # 'column'  ->  bluegreen
        [100, 100, 255],  # 'window'  ->  bright green
        [200, 200, 100],  # 'door'   ->  dark green
        [255, 0, 0],  # 'chair'  ->  darkblue
        [170, 120, 200],  # 'table'  ->  dark grey
        [10, 200, 100],  # 'bookcase'  ->  red
        [200, 100, 100],  # 'sofa'  ->  purple
        [200, 200, 200],  # 'board'   ->  grey
        [50, 50, 50],  # 'clutter'  ->  light grey
        [0, 0, 0],  # unlabelled .->. black
        [255, 255, 255],  # unlabelled .->. white
    ]
)


for fn in data_list:
    # new dirname
    new_dirname = fn.replace('/', '#').replace('.npy', '')
    new_dirpath = os.path.join(visualize_root, new_dirname)
    os.makedirs(new_dirpath, exist_ok=True)
    # load coords, rgb, labels
    coords, feats, labels, supvox = load_data(os.path.join(root, fn))
    # visualize rgb
    to_ply(coords, feats, os.path.join(new_dirpath, 'rgb.ply'))
    # visualize labels
    to_ply(coords, OBJECT_COLOR[labels], os.path.join(new_dirpath, 'gt.ply'))
    # mask out supvox 0
    labels[supvox == 0] = 14
    to_ply(coords, OBJECT_COLOR[labels], os.path.join(new_dirpath, 'better_label.ply'))
