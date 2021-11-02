"""
Data preprocessing: generate color gradient for S3DIS dataset
"""

import os
import argparse
import numpy as np
from sklearn.neighbors import KDTree
from plyfile import PlyData, PlyElement
# import matplotlib.pyplot as plt
from numba import jit, prange


# Training settings
parser = argparse.ArgumentParser(description='')
# basic
parser.add_argument('-n', '--name', choices=['s3dis', 'semantic_kitti', 'scannet'], default='s3dis',
                    help='training dataset (default: s3dis)')
parser.add_argument('-d', '--data_dir', default='/tmp2/tsunghan/S3DIS_processed/')
args = parser.parse_args()

 
root = args.data_dir
if args.name == "s3dis":
    seq_ids = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
elif args.name == "semantic_kitti":
    seq_ids = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
k_n = 50
# seq_ids = ['Area_5']


@jit('float32[:](float32[:,:],float32[:,:],int64[:,:])', nopython=True, cache=True, parallel=True)
def count_colorgrad(rgb, coords, nhoods):
    n = len(nhoods)
    out_arr = np.zeros(n, dtype=np.float32)
    for idx in prange(n):
        # diff = coords[nhoods[idx]] - coords[idx]
        # dist = diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2
        cur_rgb = rgb[idx]
        neigh_rgb = rgb[nhoods[idx]]
        diff = np.mean(np.abs(neigh_rgb - cur_rgb))
        out_arr[idx] = diff
    return out_arr


def to_ply(pos, colors, ply_fname):
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
    PlyData([el], text=False).write(ply_fname)


for seq_id in seq_ids:
    seq_dir = os.path.join(root, seq_id)
    coords_dir = os.path.join(seq_dir, 'coords')
    colorgrad_dir = os.path.join(seq_dir, 'colorgrad')
    os.makedirs(colorgrad_dir, exist_ok=True)

    for fname in os.listdir(coords_dir):
        coords_path = os.path.join(coords_dir, fname)
        rgb_path = coords_path.replace('coords', 'rgb')
        xyz = np.load(coords_path)
        tree = KDTree(xyz, leaf_size=60)
        rgb = np.load(rgb_path).astype(np.float32) / 255
        nhoods = tree.query(xyz, k=k_n, return_distance=False)
        colorgrad_npy = count_colorgrad(rgb, xyz, nhoods)
        colorgrad_npy[colorgrad_npy > 0.1] = 0.1
        # cm = plt.get_cmap('jet')
        # norm = plt.Normalize(0, 0.1)
        # entropy_color = cm(norm(colorgrad_npy)) * 255
        # to_ply(xyz, entropy_color, 'colorgrad.ply')
        # to_ply(xyz, rgb, 'rgb.ply')

        # np.save("test_colorgrad.npy", colorgrad_npy)
        # np.save("test_xyz.npy", xyz)
        # np.save("test_rgb.npy", rgb)
        new_path = os.path.join(colorgrad_dir, f'{fname[:-4]}.npy')
        np.save(new_path, colorgrad_npy)
