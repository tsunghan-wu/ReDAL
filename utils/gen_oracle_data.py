import os
import json
import numpy as np


if __name__ == "__main__":
    root = "/tmp3/tsunghan/S3DIS_processed"
    seq_ids = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']

    init_ulabel_supvox = {}
    init_label_scan = ["Area_2#hallway_8", "Area_2#WC_1", "Area_4#conferenceRoom_1", "Area_1#office_14"]
    N = 0
    for seq_id in seq_ids:
        seq_dir = os.path.join(root, seq_id)
        supvox_dir = os.path.join(seq_dir, 'supervoxel')
        for fname in os.listdir(supvox_dir):
            key = f'{seq_id}#{fname[:-4]}'
            if key in init_label_scan:
                continue
            supvox_fn = os.path.join(supvox_dir, fname)
            label_fn = supvox_fn.replace('supervoxel', 'labels')

            supvox = np.load(supvox_fn)
            label = np.load(label_fn)
            max_id = np.max(supvox)
            for i in range(1, max_id+1):
                mask = (supvox == i)
                region = label[mask]
                num_label_in_region = len(np.unique(region))
                if num_label_in_region == 1:
                    if key not in init_ulabel_supvox:
                        init_ulabel_supvox[key] = []
                    init_ulabel_supvox[key].append(i)
                    N += 1
    print(N)
    with open('init_oracle_ulabel_region.json', 'w') as f:
        json.dump(init_ulabel_supvox, f)
