"""
Generate Label/U-Labeled Initial Region-Active List
"""
import numpy as np
import argparse
import os
import json
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="Data root. Ex: ./sequences/", type=str, required=True)
    args = parser.parse_args()
    return args


def load_data(data_root, item):
    seq_id, scan_id = item
    supvox_fname = os.path.join(data_root, seq_id, "supervoxel", scan_id + ".bin")
    supvox = np.fromfile(supvox_fname, dtype=np.int32).reshape(-1)
    return supvox


if __name__ == "__main__":
    args = get_args()
    # Get items list
    seq_lst = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    # seq_lst = ['02']
    items = []

    for seq_id in seq_lst:
        seq_dir = os.path.join(args.data_root, seq_id, "velodyne")
        for item in os.listdir(seq_dir):
            scan_id = os.path.splitext(item)[0]
            info = (seq_id, scan_id)
            items.append(info)

    supvox_list = []
    for item in tqdm(items):
        seq_id, scan_id = item
        # Load data
        supvox = load_data(args.data_root, item)
        num_seg = np.max(supvox)
        for i in range(1, num_seg + 1):
            k = (seq_id, scan_id, i)
            supvox_list.append(k)
    # Random Shuffle List
    np.random.shuffle(supvox_list)
    N = len(supvox_list) // 100
    labeled_lst = supvox_list[:N]
    ulabeled_lst = supvox_list[N:]

    # Save out
    labeled_json = {}
    ulabeled_json = {}

    for item in tqdm(labeled_lst):
        seq_id, scan_id, seg_id = item
        k = seq_id + "_" + scan_id
        if k not in labeled_json:
            labeled_json[k] = []
        labeled_json[k].append(seg_id)

    for item in tqdm(ulabeled_lst):
        seq_id, scan_id, seg_id = item
        k = seq_id + "_" + scan_id
        if k not in ulabeled_json:
            ulabeled_json[k] = []
        ulabeled_json[k].append(seg_id)

    with open("region_label_data.json", "w") as f:
        json.dump(labeled_json, f)
    with open("region_ulabel_data.json", "w") as f:
        json.dump(ulabeled_json, f)
