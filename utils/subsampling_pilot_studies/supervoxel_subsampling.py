"""
Oracle Supervoxel uncertain-based subsampling (now useless)
"""

import os
from tqdm import tqdm
import numpy as np
from plyfile import PlyData, PlyElement


root_dir = '/tmp3/tsunghan/S3DIS_processed'
subsampling_dir = 'ent_select'
new_root_dir = '/tmp3/tsunghan/S3DIS_subsample'
os.makedirs(new_root_dir, exist_ok=True)
train_split = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']


def transform_fn(arr):
    return arr ** (0.1)


def save_ply(pos, colors, ply_fname):
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
    PlyData([el], text=False).write(ply_fname)


for active_selection in tqdm(range(1, 10)):
    selection_root_dir = os.path.join(new_root_dir, f'selection_{active_selection}')
    os.makedirs(selection_root_dir, exist_ok=True)
    for split in train_split:
        rgb_dirname = os.path.join(root_dir, split, 'rgb')
        labels_dirname = os.path.join(root_dir, split, 'labels')
        coords_dirname = os.path.join(root_dir, split, 'coords')
        supvox_dirname = os.path.join(root_dir, split, 'supervoxel')

        new_rgb_dirname = os.path.join(selection_root_dir, split, 'rgb')
        new_labels_dirname = os.path.join(selection_root_dir, split, 'labels')
        new_coords_dirname = os.path.join(selection_root_dir, split, 'coords')
        new_ply_dirname = os.path.join(selection_root_dir, split, 'ply')
        os.makedirs(new_rgb_dirname, exist_ok=True)
        os.makedirs(new_labels_dirname, exist_ok=True)
        os.makedirs(new_coords_dirname, exist_ok=True)
        os.makedirs(new_ply_dirname, exist_ok=True)

        for fname in os.listdir(coords_dirname):
            rgb_fname = os.path.join(rgb_dirname, fname)
            labels_fname = os.path.join(labels_dirname, fname)
            coords_fname = os.path.join(coords_dirname, fname)
            supvox_fname = os.path.join(supvox_dirname, fname)

            coords = np.load(coords_fname)
            rgb = np.load(rgb_fname)
            labels = np.load(labels_fname)
            supvox = np.load(supvox_fname)

            labels_ = labels.reshape(-1)

            basename = f'{split}#coords#{fname}'
            entropy_fname = os.path.join(subsampling_dir, f'selection_{active_selection}', basename)
            if os.path.isfile(entropy_fname):
                entropy = np.load(entropy_fname)
                entropy = transform_fn(entropy)
                prob = np.random.rand(entropy.shape[0])
                preserving_idx = (prob <= entropy)
                sub_rgb = rgb[preserving_idx]
                sub_coords = coords[preserving_idx]
                sub_labels = labels[preserving_idx]

                new_rgb_fname = os.path.join(new_rgb_dirname, fname)
                new_labels_fname = os.path.join(new_labels_dirname, fname)
                new_coords_fname = os.path.join(new_coords_dirname, fname)
                new_ply_fname = os.path.join(new_ply_dirname, fname[:-4]+'.ply')

                np.save(new_rgb_fname, sub_rgb)
                np.save(new_coords_fname, sub_coords)
                np.save(new_labels_fname, sub_labels)

                save_ply(sub_coords, sub_rgb, new_ply_fname)
