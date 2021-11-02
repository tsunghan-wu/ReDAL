# Copy from SparseConvNet by Ben Graham
# https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py

import argparse
import glob
import plyfile
import numpy as np
import multiprocessing as mp
import os
# import torch

parser = argparse.ArgumentParser()
parser.add_argument('--num-cpu', type=int, default=1, help='How many cpu thread to use')
parser.add_argument('--data-root', required=True, help='path to the all the scan files')
args = parser.parse_args()

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i


files = sorted(glob.glob(args.data_root + '*/*_vh_clean_2.ply'))
files2 = sorted(glob.glob(args.data_root + '*/*_vh_clean_2.labels.ply'))

assert len(files) == len(files2)
assert len(files) > 0
print('Total', len(files), 'files')


def f(fn):
    fn2 = fn[:-3] + 'labels.ply'
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, 3:6])
    a = plyfile.PlyData().read(fn2)
    w = remapper[np.array(a.elements[0]['label'])]
    # print(fn, fn2)
    parent_dir = os.path.abspath(os.path.join(fn, os.pardir))
    labels_fname = os.path.join(parent_dir, 'labels.npy')
    coords_fname = os.path.join(parent_dir, 'coords.npy')
    rgb_fname = os.path.join(parent_dir, 'rgb.npy')
    np.save(coords_fname, coords)
    np.save(labels_fname, w)
    np.save(rgb_fname, colors)


ncpu = min(args.num_cpu, mp.cpu_count())
print(f'Using: {ncpu} CPUs')
p = mp.Pool(processes=ncpu)
p.map(f, files)
p.close()
p.join()
