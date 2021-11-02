"""
Oracle Random Subsampling pilot studies (now useless)
"""

import os
import numpy as np

sample_rate = 0.1
root_dir = '/work/patrickwu2/S3DIS_processed'
train_split = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
new_root_dir = f'{root_dir}_{sample_rate}'
os.makedirs(new_root_dir, exist_ok=True)


for split in train_split:
    rgb_dirname = os.path.join(root_dir, split, 'rgb')
    labels_dirname = os.path.join(root_dir, split, 'labels')
    coords_dirname = os.path.join(root_dir, split, 'coords')

    new_rgb_dirname = os.path.join(new_root_dir, split, 'rgb')
    new_labels_dirname = os.path.join(new_root_dir, split, 'labels')
    new_coords_dirname = os.path.join(new_root_dir, split, 'coords')
    os.makedirs(new_rgb_dirname, exist_ok=True)
    os.makedirs(new_labels_dirname, exist_ok=True)
    os.makedirs(new_coords_dirname, exist_ok=True)

    for fname in os.listdir(coords_dirname):
        rgb_fname = os.path.join(rgb_dirname, fname)
        labels_fname = os.path.join(labels_dirname, fname)
        coords_fname = os.path.join(coords_dirname, fname)

        coords = np.load(coords_fname)
        rgb = np.load(rgb_fname)
        labels = np.load(labels_fname)

        labels_ = labels.reshape(-1)

        arg = np.argwhere((labels_ == 0) | (labels_ == 1))[:, 0]    # ceiling & floor
        np.random.shuffle(arg)
        total_points = arg.shape[0]
        delete_idx = arg[:(total_points * 9 // 10)]

        sub_rgb = np.delete(rgb, delete_idx, axis=0)
        sub_coords = np.delete(coords, delete_idx, axis=0)
        sub_labels = np.delete(labels, delete_idx, axis=0)

        new_rgb_fname = os.path.join(new_root_dir, split, 'rgb', fname)
        new_labels_fname = os.path.join(new_root_dir, split, 'labels', fname)
        new_coords_fname = os.path.join(new_root_dir, split, 'coords', fname)

        np.save(new_rgb_fname, sub_rgb)
        np.save(new_coords_fname, sub_coords)
        np.save(new_labels_fname, sub_labels)
