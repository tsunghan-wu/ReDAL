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
    parser.add_argument("--data_root", help="Data root. Ex: ./S3DIS_processed/", type=str, required=True)
    parser.add_argument("--json_file", help="Source json file. Ex: ./label_data.json", type=str, required=True)
    parser.add_argument("--output", help="Destination json file. Ex: ./region_label_data.json", type=str, required=True)
    args = parser.parse_args()
    return args


def load_data(data_root, item):
    scan_id = item
    supvox_fname = os.path.join(data_root, scan_id, "supervoxel.npy")
    supvox = np.load(supvox_fname).reshape(-1)
    return supvox


if __name__ == "__main__":
    args = get_args()
    # Get items list
    with open(args.json_file, 'r') as f:
        fnames = json.load(f)
    items = []
    for fname in fnames:
        scan_id, _ = fname.split('/')
        items.append(scan_id)
    supvox_list = []
    for item in tqdm(items):
        scan_id = item
        # Load data
        supvox = load_data(args.data_root, item)
        num_seg = np.max(supvox)
        for i in range(1, num_seg + 1):
            k = (scan_id, i)
            supvox_list.append(k)
    # Save out
    output_json = {}

    for item in tqdm(supvox_list):
        scan_id, seg_id = item
        k = scan_id
        if k not in output_json:
            output_json[k] = []
        output_json[k].append(seg_id)
    with open(args.output, "w") as f:
        json.dump(output_json, f)
