"""
Oracle Grid Subsampling (Now useless)
"""

import os
import numpy as np
import utils.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

sub_grid_size = 0.1
root_dir = '/work/patrickwu2/S3DIS_processed'
train_split = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
new_root_dir = f'{root_dir}_{sub_grid_size}'
os.makedirs(new_root_dir, exist_ok=True)


def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
    """
    CPP wrapper for a grid sub_sampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param grid_size: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: sub_sampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
    elif labels is None:
        return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
    elif features is None:
        return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                       verbose=verbose)


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

        sub_coords, sub_rgb, sub_labels = grid_sub_sampling(coords, rgb, labels, sub_grid_size)

        new_rgb_fname = os.path.join(new_root_dir, split, 'rgb', fname)
        new_labels_fname = os.path.join(new_root_dir, split, 'labels', fname)
        new_coords_fname = os.path.join(new_root_dir, split, 'coords', fname)

        np.save(new_rgb_fname, sub_rgb)
        np.save(new_coords_fname, sub_coords)
        np.save(new_labels_fname, sub_labels)
