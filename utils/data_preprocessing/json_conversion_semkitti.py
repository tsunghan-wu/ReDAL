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
    parser.add_argument("--json_file", help="Source json file. Ex: ./label_data.json", type=str, required=True)
    parser.add_argument("--output", help="Destination json file. Ex: ./region_label_data.json", type=str, required=True)
    args = parser.parse_args()
    return args


def load_data(data_root, item):
    seq_id, scan_id = item
    supvox_fname = os.path.join(data_root, seq_id, "supervoxel_large", scan_id + ".bin")
    supvox = np.fromfile(supvox_fname, dtype=np.int32).reshape(-1)
    return supvox


if __name__ == "__main__":
    args = get_args()
    # Get items list
    with open(args.json_file, 'r') as f:
        fnames = json.load(f)
    items = []
    for fname in fnames:
        seq_id, _, scan_fname = fname.split('/')
        scan_id = scan_fname[:-4]
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
    # Save out
    output_json = {}

    for item in tqdm(supvox_list):
        seq_id, scan_id, seg_id = item
        k = seq_id + "_" + scan_id
        if k not in output_json:
            output_json[k] = []
        output_json[k].append(seg_id)
    with open(args.output, "w") as f:
        json.dump(output_json, f)
