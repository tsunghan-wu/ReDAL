import os
import pickle
import numpy as np
from dataloader.scannet.region_dataset import RegionScannetDataset


class RegionActiveScannet:
    def __init__(self, args):
        # Active Learning intitial selection
        self.args = args
        self.selection_iter = 0
        self.label_dataset = RegionScannetDataset(args.data_dir, imageset='active-label', voxel_size=0.05)
        self.pool_dataset = RegionScannetDataset(args.data_dir, imageset='active-ulabel', voxel_size=0.05)

    def expand_training_set(self, sample_region, percent):
        """
        Parameter: sample_region (list)
        [
            (score, scan_file_path, supvox_id),
            ...
        ]
        """
        max_selection_count = int(self.total * percent / 100)
        selected_count = 0

        for idx, x in enumerate(sample_region):
            _, scan_file_path, supvox_id = x
            fn = scan_file_path.replace('coords', 'supervoxel')
            supvox = np.load(fn)
            selected_count += (supvox == supvox_id).sum()
            # Add into label dataset
            if scan_file_path not in self.label_dataset.im_idx:
                self.label_dataset.im_idx.append(scan_file_path)
                self.label_dataset.supvox[scan_file_path] = [supvox_id]
            else:
                self.label_dataset.supvox[scan_file_path].append(supvox_id)
            # Remove it from unlabeled dataset
            self.pool_dataset.supvox[scan_file_path].remove(supvox_id)
            if len(self.pool_dataset.supvox[scan_file_path]) == 0:
                self.pool_dataset.supvox.pop(scan_file_path)
                self.pool_dataset.im_idx.remove(scan_file_path)
            # jump out the loop when exceeding max_selection_count
            if selected_count > max_selection_count:
                selection_path = os.path.join(self.args.model_save_dir, f'selection_{self.selection_iter:02d}.pkl')
                with open(selection_path, "wb") as f:
                    pickle.dump(sample_region[:idx+1], f)
                break

    def get_fraction_of_labeled_data(self):
        self.total = 0
        label_num = 0
        for key in self.label_dataset.supvox:
            fn = key.replace('coords', 'supervoxel')
            supvox = np.load(fn)
            preserving_labels = self.label_dataset.supvox[key]
            mask = np.isin(supvox, preserving_labels)
            label_num += mask.sum()
        pool_num = 0
        for key in self.pool_dataset.supvox:
            fn = key.replace('coords', 'supervoxel')
            supvox = np.load(fn)
            preserving_labels = self.pool_dataset.supvox[key] + [0]
            mask = np.isin(supvox, preserving_labels)
            pool_num += mask.sum()
        self.total = label_num + pool_num
        return label_num / (label_num + pool_num)

    def get_number_of_labeled_region(self):
        label_num = 0
        for key in self.label_dataset.supvox:
            label_num += len(self.label_dataset.supvox[key])
        return label_num

    def dump_datalist(self):
        datalist_path = os.path.join(self.args.model_save_dir, f'datalist_{self.selection_iter:02d}.pkl')
        with open(datalist_path, "wb") as f:
            store_data = {
                'L_im_idx': self.label_dataset.im_idx,
                'U_im_idx': self.pool_dataset.im_idx,
                'L_supvox': self.label_dataset.supvox,
                'U_supvox': self.pool_dataset.supvox
            }
            pickle.dump(store_data, f)

    def load_datalist(self, convert_root=False):
        print('Load path', flush=True)
        # Synchronize Training Path
        datalist_path = os.path.join(self.args.model_save_dir, f'datalist_{self.selection_iter:02d}.pkl')
        with open(datalist_path, "rb") as f:
            pickle_data = pickle.load(f)
        if convert_root is True:
            pickle_data = convert_root_fn(pickle_data, self.args.data_dir)
        self.label_dataset.im_idx = pickle_data['L_im_idx']
        self.pool_dataset.im_idx = pickle_data['U_im_idx']
        self.label_dataset.supvox = pickle_data['L_supvox']
        self.pool_dataset.supvox = pickle_data['U_supvox']


def convert_root_fn(pickle_data, root_dir):
    new_dict = {}
    # L_im_idx / U_im_idx
    new_dict['L_im_idx'] = []
    new_dict['L_supvox'] = {}
    for path in pickle_data['L_im_idx']:
        supvox_lst = pickle_data['L_supvox'][path]
        basename = '/'.join(path.split('/')[-3:])
        new_path = os.path.join(root_dir, basename)
        new_dict['L_im_idx'].append(new_path)
        new_dict['L_supvox'][new_path] = supvox_lst
    new_dict['U_im_idx'] = []
    new_dict['U_supvox'] = {}

    for path in pickle_data['U_im_idx']:
        supvox_lst = pickle_data['U_supvox'][path]
        basename = '/'.join(path.split('/')[-3:])
        new_path = os.path.join(root_dir, basename)
        new_dict['U_im_idx'].append(new_path)
        new_dict['U_supvox'][new_path] = supvox_lst
    return new_dict
