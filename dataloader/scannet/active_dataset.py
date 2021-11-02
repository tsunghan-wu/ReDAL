import os
import json
from dataloader.scannet.dataset import ScannetDataset


class ActiveScannetDataset:
    def __init__(self, args):
        # Active Learning intitial selection
        self.args = args
        self.selection_iter = 0
        self.label_dataset = ScannetDataset(args.data_dir, imageset='active-label', voxel_size=0.05)
        self.pool_dataset = ScannetDataset(args.data_dir, imageset='active-ulabel', voxel_size=0.05)

    def expand_training_set(self, paths):
        self.label_dataset.im_idx.extend(paths)
        for x in paths:
            self.pool_dataset.im_idx.remove(x)

    def get_fraction_of_labeled_data(self):
        label_num = len(self.label_dataset.im_idx)
        pool_num = len(self.pool_dataset.im_idx)
        return label_num / (label_num + pool_num)

    def dump_datalist(self):
        datalist_path = os.path.join(self.args.model_save_dir, f'datalist_{self.selection_iter:02d}.json')
        with open(datalist_path, "w") as f:
            store_data = {
                'L_im_idx': self.label_dataset.im_idx,
                'U_im_idx': self.pool_dataset.im_idx,
            }
            json.dump(store_data, f)

    def load_datalist(self, convert_root=False):
        print('Load path', flush=True)
        # Synchronize Training Path
        datalist_path = os.path.join(self.args.model_save_dir, f'datalist_{self.selection_iter:02d}.json')
        with open(datalist_path, "rb") as f:
            json_data = json.load(f)
        if convert_root is True:
            json_data = convert_root_fn(json_data, self.args.data_dir)
        self.label_dataset.im_idx = json_data['L_im_idx']
        self.pool_dataset.im_idx = json_data['U_im_idx']


def convert_root_fn(pickle_data, root_dir):
    new_dict = {}
    # L_im_idx / U_im_idx
    new_dict['L_im_idx'] = []
    for path in pickle_data['L_im_idx']:
        basename = '/'.join(path.split('/')[-3:])
        new_path = os.path.join(root_dir, basename)
        new_dict['L_im_idx'].append(new_path)
    new_dict['U_im_idx'] = []

    for path in pickle_data['U_im_idx']:
        basename = '/'.join(path.split('/')[-3:])
        new_path = os.path.join(root_dir, basename)
        new_dict['U_im_idx'].append(new_path)
    return new_dict
