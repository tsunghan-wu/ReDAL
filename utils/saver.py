import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement


class Saver:
    def __init__(self, args):
        self.dataset_name = args.name
        self.args = args

    def load_data(self, fn):
        '''
        Return coords, rgb, labels, supvox
        '''
        raise NotImplementedError("Need to implement")

    def load_data_list(self):
        '''
        Return:
            - data_list(List)
            - original_selections(dict)
        '''
        data_list = []
        original_selections = {}
        init_data_directory = f'dataloader/{self.dataset_name}/init_data'
        init_lscan = os.path.join(init_data_directory, 'init_label_scan.json')
        init_uscan = os.path.join(init_data_directory, 'init_ulabel_scan.json')
        init_lregion = os.path.join(init_data_directory, 'init_label_region.json')
        # read init data list
        with open(init_lscan, 'r') as f:
            data_list.extend(json.load(f))
        with open(init_uscan, 'r') as f:
            data_list.extend(json.load(f))

        # read init selection
        for fname in data_list:
            original_selections[fname] = []
        with open(init_lregion, 'r') as f:
            region_dict = json.load(f)
        for key in region_dict:
            seq_id, scan_id = key.split('#')
            new_key = f'{seq_id}/{self.coords_dirname}/{scan_id}.{self.ext}'
            original_selections[new_key] = region_dict[key]
        return data_list, original_selections

    @staticmethod
    def to_ply(pos, colors, ply_fname):
        # to ply
        ply_array = np.ones(
            pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        )
        ply_array["x"] = pos[:, 0]
        ply_array["y"] = pos[:, 1]
        ply_array["z"] = pos[:, 2]
        ply_array["red"] = colors[:, 0]
        ply_array["green"] = colors[:, 1]
        ply_array["blue"] = colors[:, 2]
        el = PlyElement.describe(ply_array, 'vertex')
        PlyData([el], text=False).write(ply_fname)

    def save_input_gt_sup(self):
        for fname in self.data_list:
            abs_path = os.path.join(self.args.data_dir, fname)
            coords, rgb, labels, supvox = self.load_data(abs_path)

            basename = fname.replace("/", "#")[:-4]
            save_dir = os.path.join(self.args.output_dir, basename)
            os.makedirs(save_dir, exist_ok=True)

            # save rgb
            self.to_ply(coords, rgb, os.path.join(save_dir, 'rgb.ply'))
            # save gt
            self.to_ply(coords, self.LABEL_TO_COLOR[labels], os.path.join(save_dir, 'gt.ply'))
            # save supervoxel
            vals = np.linspace(0, 1, np.max(supvox))
            np.random.shuffle(vals)
            cm = plt.cm.colors.ListedColormap(plt.cm.hsv(vals))
            # cm = plt.cm.get_cmap('hsv', np.max(supvox))
            self.to_ply(coords, cm(supvox) * 255, os.path.join(save_dir, 'supvox.ply'))

    def save_pred_error_entropy(self, coords, pred, gt, entropy, fname, AL_iter):
        basename = '#'.join(fname.split('/')[-3:])[:-4]
        save_dir = os.path.join(self.args.output_dir, basename, f'selection_{AL_iter}')
        os.makedirs(save_dir, exist_ok=True)
        # save pred
        self.to_ply(coords, self.LABEL_TO_COLOR[pred], os.path.join(save_dir, 'pred.ply'))
        # save error
        mask = (pred != gt)
        np.save(os.path.join(save_dir, 'error.npy'), mask)
        error_color = np.ones_like(coords).astype(np.uint8) * 255
        error_color[mask] = np.array([[255, 0, 0]])
        self.to_ply(coords, error_color, os.path.join(save_dir, 'error.ply'))
        # save entropy
        cm = plt.get_cmap('jet')
        norm = plt.Normalize(0, 0.25)
        entropy_color = cm(norm(entropy)) * 255
        self.to_ply(coords, entropy_color, os.path.join(save_dir, 'entropy.ply'))
        np.save(os.path.join(save_dir, 'entropy.npy'), entropy)

    def save_pred_error(self, coords, pred, gt, fname, AL_iter):
        basename = '#'.join(fname.split('/')[-3:])[:-4]
        save_dir = os.path.join(self.args.output_dir, basename, f'selection_{AL_iter}')
        os.makedirs(save_dir, exist_ok=True)
        # save pred
        self.to_ply(coords, self.LABEL_TO_COLOR[pred], os.path.join(save_dir, 'pred.ply'))
        # save error
        mask = (pred != gt)
        np.save(os.path.join(save_dir, 'error.npy'), mask)
        error_color = np.ones_like(coords).astype(np.uint8) * 255
        error_color[mask] = np.array([[255, 0, 0]])
        self.to_ply(coords, error_color, os.path.join(save_dir, 'error.ply'))

    def save_selections(self, log_dir, AL_iter):
        new_selections = self.parse_selection_file(log_dir, AL_iter)

        for fname in self.data_list:
            fn = os.path.join(self.args.data_dir, fname)
            original_supvox_lst = self.original_selections[fname]
            new_supvox_lst = []
            if fname in new_selections:
                new_supvox_lst = new_selections[fname]
            basename = fname.replace("/", "#")[:-4]
            save_dir = os.path.join(self.args.output_dir, basename, f'selection_{AL_iter}')
            os.makedirs(save_dir, exist_ok=True)
            self.write_selections(fn, new_supvox_lst, original_supvox_lst, save_dir)
            # Merge lst
            self.original_selections[fname].extend(new_supvox_lst)

    def write_selections(self, fn, current_supvox_id, original_supvox_id, save_dir):
        # Load data
        coords, rgb, labels, supvox = self.load_data(fn)

        # original mask
        original_supvox = labels.copy()
        mask = np.isin(supvox, original_supvox_id)
        original_supvox = np.where(mask, original_supvox, self.ignore_label)
        self.to_ply(coords, self.LABEL_TO_COLOR[original_supvox], os.path.join(save_dir, 'original.ply'))

        # current mask
        mask = np.isin(supvox, current_supvox_id)
        current_supvox = np.where(mask, labels, self.ignore_label)
        self.to_ply(coords, self.LABEL_TO_COLOR[current_supvox], os.path.join(save_dir, 'select.ply'))

    def parse_selection_file(self, log_dir, selection_iter):
        selection_file = os.path.join(log_dir, f'selection_{selection_iter:02d}.pkl')
        with open(selection_file, 'rb') as f:
            selections = pickle.load(f)
        supvox = {}
        for item in selections:
            _, scan_file_path, supvox_id = item
            raw_path = '/'.join(scan_file_path.split('/')[-3:])
            if raw_path not in supvox:
                supvox[raw_path] = []
            supvox[raw_path].append(supvox_id)
        return supvox
