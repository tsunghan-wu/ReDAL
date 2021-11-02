"""
Data preprocessing: generate KDTree for S3DIS dataset
"""

import os
import pickle
import numpy as np
from sklearn.neighbors import KDTree


root = "/tmp3/tsunghan/S3DIS_processed"

# seq_ids = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
seq_ids = ['Area_5']

for seq_id in seq_ids:
    seq_dir = os.path.join(root, seq_id)
    coords_dir = os.path.join(seq_dir, 'coords')
    kd_dir = os.path.join(seq_dir, 'kdtree')
    os.makedirs(kd_dir, exist_ok=True)

    for fname in os.listdir(coords_dir):
        coords_path = os.path.join(coords_dir, fname)
        xyz = np.load(coords_path)

        tree = KDTree(xyz, leaf_size=60)
        new_path = os.path.join(kd_dir, f'{fname[:-4]}.pkl')
        with open(new_path, "wb") as f:
            pickle.dump(tree, f)
