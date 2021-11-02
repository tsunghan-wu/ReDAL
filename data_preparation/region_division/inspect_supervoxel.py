import numpy as np
import argparse
import os
import yaml
from tqdm import tqdm


class SuperVoxelAnalyzor:
    def __init__(self):
        self.seg_points = []    # Record number of points in a segment
        self.seg_purity = []    # Record purity of a segment
        self.seg_volume = []    # Record number of occupied voxel (volume)

    def cal_points(self, velo, supvox, label, idx_mask):
        self.seg_points.append(len(idx_mask))
        return

    def cal_purity(self, velo, supvox, label, idx_mask):
        region_label = label[idx_mask]
        num_label_in_region = len(np.unique(region_label))
        self.seg_purity.append(num_label_in_region)
        return

    def cal_volume(self, velo, supvox, label, idx_mask):
        region_points = velo[idx_mask]
        min_value = np.min(region_points, axis=0)
        max_value = np.max(region_points, axis=0)
        diff = max_value - min_value
        vox = diff[0] * diff[1]
        self.seg_volume.append(vox)
        return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="Data root. Ex: ./sequences/", type=str, required=True)
    parser.add_argument("--yaml_file", help="yaml file. Ex: semantic-kitti.yaml", type=str, required=True)
    args = parser.parse_args()
    return args


def load_data(data_root, item, learning_map):
    seq_id, scan_id = item
    velo_fname = os.path.join(data_root, seq_id, "velodyne", scan_id + ".bin")
    supvox_fname = os.path.join(data_root, seq_id, "supervoxel", scan_id + ".bin")
    label_fname = os.path.join(data_root, seq_id, "labels", scan_id + ".label")

    # Read data
    velo = np.fromfile(velo_fname, dtype=np.float32).reshape(-1, 4)[:, :3]
    supvox = np.fromfile(supvox_fname, dtype=np.int32).reshape(-1)
    label = np.fromfile(label_fname, dtype=np.int32).reshape(-1)
    label = label & 0xFFFF
    label = np.vectorize(learning_map.__getitem__)(label)
    return velo, supvox, label


if __name__ == "__main__":
    args = get_args()
    # Load yaml
    with open(args.yaml_file, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    learning_map = semkittiyaml['learning_map']
    # Get items list
    seq_lst = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    items = []

    for seq_id in seq_lst:
        seq_dir = os.path.join(args.data_root, seq_id, "velodyne")
        for item in os.listdir(seq_dir):
            scan_id = os.path.splitext(item)[0]
            info = (seq_id, scan_id)
            items.append(info)
    items.sort()
    cls = SuperVoxelAnalyzor()
    for item in tqdm(items):
        # Load data
        velo, supvox, label = load_data(args.data_root, item, learning_map)
        num_seg = np.max(supvox)
        for seg_id in range(1, num_seg+1):
            idx_mask = (supvox == seg_id).nonzero()[0]
            cls.cal_points(velo, supvox, label, idx_mask)   # count number of points in a region
            cls.cal_purity(velo, supvox, label, idx_mask)   # calculate region purity
            cls.cal_volume(velo, supvox, label, idx_mask)   # calculate region volume
    np.savetxt("new_purity.csv", cls.seg_purity, delimiter=',', fmt='%i')
    np.savetxt("all_volume.csv", cls.seg_volume, delimiter=',')
    np.savetxt("mass.csv", cls.seg_points, delimiter=',')
    np.savetxt("density.csv", cls.seg_points / cls.seg_volume, delimiter=',')
