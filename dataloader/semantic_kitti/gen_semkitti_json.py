import os
import json
import numpy as np
from tqdm import tqdm

data_root = "/work/patrickwu2/SemanticKitti/sequences"
seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']

save_json = {}

for seq in tqdm(seqs):
    seq_dir = os.path.join(data_root, seq)
    coords_dir = os.path.join(seq_dir, "velodyne")
    supvox_dir = os.path.join(seq_dir, "supervoxel_large")
    for fname in os.listdir(coords_dir):
        coords_fname = os.path.join(coords_dir, fname)
        supvox_fname = os.path.join(supvox_dir, fname)

        coords = np.fromfile(coords_fname, dtype=np.float32).reshape(-1, 4)
        supvox = np.fromfile(supvox_fname, dtype=np.int32).reshape(-1)

        fname = '/'.join(coords_fname.split('/')[-3:])
        max_id = supvox.max()
        for i in range(1, max_id + 1):
            key = f'{fname}#{i}'
            pts = (supvox == i).sum()
            save_json[key] = pts
            # print(key)
            # exit()
with open("semkitti_large_pts.json", "w") as f:
    json.dump(save_json, f)
