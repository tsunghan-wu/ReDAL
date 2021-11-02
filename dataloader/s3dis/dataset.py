import os
import glob
import json
import pickle
import numpy as np

import dataloader.s3dis.transforms as t
from torchsparse.utils import sparse_collate_fn, sparse_quantize
from torchsparse import SparseTensor


class Stanford3DDataset:
    NUM_CLASSES = 13
    CLIP_BOUND = 4
    TEST_CLIP_BOUND = None
    IGNORE_LABEL = -100
    ROTATION_AXIS = 'z'

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = \
        ((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (-0.05, 0.05))

    TRAIN_AREA_IDS = [1, 2, 3, 4, 6]
    TEST_AREA_IDS = [5]

    def __init__(self, data_root, voxel_size, imageset='train', init_lst=None):
        # ASSIGN PARAMETERS
        self.data_root = data_root
        self.voxel_size = voxel_size

        self.imageset = imageset
        if imageset == 'train':
            split = self.TRAIN_AREA_IDS
            self.use_augs = {
                'scale': True, 'rotate': True, 'elastic': True, 'chromatic': True
            }
        elif imageset == 'val':
            split = self.TEST_AREA_IDS
            self.use_augs = {
                'scale': False, 'rotate': False, 'elastic': False, 'chromatic': False
            }
        elif imageset == 'test':
            split = self.TEST_AREA_IDS
            self.use_augs = {
                'scale': False, 'rotate': False, 'elastic': False, 'chromatic': False
            }
        elif imageset == 'active-label':
            with open('dataloader/s3dis/init_data/init_label_scan.json') as f:
                lst = json.load(f)
            self.use_augs = {
                'scale': True, 'rotate': True, 'elastic': True, 'chromatic': True
            }
        elif imageset == 'active-ulabel':
            with open('dataloader/s3dis/init_data/init_ulabel_scan.json') as f:
                lst = json.load(f)
            self.use_augs = {
                'scale': False, 'rotate': False, 'elastic': False, 'chromatic': False
            }
        elif imageset == 'custom-set':
            lst = init_lst
            self.use_augs = {
                'scale': False, 'rotate': False, 'elastic': False, 'chromatic': False
            }
        if self.imageset not in ['train', 'active-label']:
            for key, val in self.use_augs.items():
                assert val is False, f"{key} should be False during evaluation"

        self.im_idx = []
        # e.g. DATA_ROOT/Area_1
        if self.imageset in ['train', 'val', 'test']:
            for x in split:
                # READ SPLIT FILES
                fn = os.path.join(self.data_root, f"Area_{x}", "coords") + '/*.npy'
                self.im_idx.extend(glob.glob(fn))
        elif imageset in ['active-label', 'active-ulabel']:
            self.im_idx = [os.path.join(self.data_root, i) for i in lst]
        elif imageset == 'custom-set':
            self.im_idx = lst
        # self.im_idx.sort()
        self.prevoxel_aug_func = self.build_prevoxel_aug_func()
        self.postvoxel_aug_func = self.build_postvoxel_aug_func()
        self.return_supvox = False
        self.load_kdtree = False

    def set_load_kdtree(self, flag):
        assert flag in [True, False]
        self.load_kdtree = flag
        print(f'Set Dataset Load KDTREE = {self.load_kdtree}', flush=True)

    def __getitem__(self, idx):
        # Read data
        if self.return_supvox is False:
            coords, feats, labels = self.load_data(self.im_idx[idx])
        elif self.return_supvox is True:
            coords, feats, supvox = self.load_supvox_data(self.im_idx[idx])
            labels = supvox

        coords = coords.astype(np.float32)
        feats = feats.astype(np.float32)
        labels = labels.astype(np.int32)

        # Prevoxel Augmentation
        if self.prevoxel_aug_func is not None:
            coords, feats, labels = self.prevoxel_aug_func(coords, feats, labels)

        # Voxelize
        pc_ = np.round(coords / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        # Postvoxel transformation
        if self.postvoxel_aug_func is not None:
            pc_, feats, labels = self.postvoxel_aug_func(pc_, feats, labels)

        labels = labels.reshape(-1)
        labels_ = labels
        feats /= 255.0
        feat_ = np.concatenate([feats, coords], axis=1)

        # Sparse Quantize
        inds, labels, inverse_map = sparse_quantize(pc_, feat_, labels_, return_index=True, return_invs=True)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)
        if self.load_kdtree is False:
            return {
                'lidar': lidar,
                'targets': labels,
                'targets_mapped': labels_,
                'inverse_map': inverse_map,
                'file_name': self.im_idx[idx]
            }
        else:
            kd_tree_path = self.im_idx[idx].replace('coords', 'kdtree').replace('npy', 'pkl')
            with open(kd_tree_path, 'rb') as f:
                tree = pickle.load(f)
            return {
                'lidar': lidar,
                'targets': labels,
                'targets_mapped': labels_,
                'inverse_map': inverse_map,
                'file_name': self.im_idx[idx],
                'kd_tree': tree
            }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)

    def label_to_supvox(self):
        self.return_supvox = True

    def supvox_to_label(self):
        self.return_supvox = False

    def load_data(self, fn):
        coords_fn = fn
        feats_fn = fn.replace('coords', 'rgb')
        labels_fn = fn.replace('coords', 'labels')
        coords = np.load(coords_fn).astype(np.float32)
        feats = np.load(feats_fn).astype(np.float32)
        labels = np.load(labels_fn).astype(np.int32)
        return coords, feats, labels

    def load_supvox_data(self, fn):
        coords_fn = fn
        feats_fn = fn.replace('coords', 'rgb')
        supvox_fn = fn.replace('coords', 'supervoxel')
        coords = np.load(coords_fn).astype(np.float32)
        feats = np.load(feats_fn).astype(np.float32)
        supvox = np.load(supvox_fn).astype(np.int32)
        return coords, feats, supvox

    def __len__(self):
        return len(self.im_idx)

    def build_prevoxel_aug_func(self):
        aug_funcs = []

        if self.use_augs.get('elastic', False):
            aug_funcs.append(
                t.RandomApply([
                    t.ElasticDistortion([(0.2, 0.4), (0.8, 1.6)])
                ], 0.95)
            )
        if self.use_augs.get('rotate', False):
            aug_funcs += [
                t.Random360Rotate(self.ROTATION_AXIS, around_center=True),
                t.RandomApply([
                    t.RandomRotateEachAxis([(-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (0, 0)])
                ], 0.95)
            ]
        if self.use_augs.get('scale', False):
            aug_funcs.append(
                t.RandomApply([t.RandomScale(0.9, 1.1)], 0.95)
            )
        if self.use_augs.get('translate', False):
            # Positive translation should do at the end. Otherwise, the coords might be in negative space
            aug_funcs.append(
                t.RandomApply([
                    t.RandomPositiveTranslate([0.2, 0.2, 0])
                ], 0.95)
            )
        if len(aug_funcs) > 0:
            return t.Compose(aug_funcs)
        else:
            return None

    def build_postvoxel_aug_func(self):
        aug_funcs = []
        if self.use_augs.get('dropout', False):
            aug_funcs.append(
                t.RandomApply([t.RandomDropout(0.2)], 0.5),
            )
        if self.use_augs.get('hflip', False):
            aug_funcs.append(
                t.RandomApply([t.RandomHorizontalFlip(self.ROTATE_AXIS)], 0.95),
            )
        if self.use_augs.get('chromatic', False):
            # The feats input should be in [0-255]
            aug_funcs += [
                t.RandomApply([t.ChromaticAutoContrast()], 0.2),
                t.RandomApply([t.ChromaticTranslation(0.1)], 0.95),
                t.RandomApply([t.ChromaticJitter(0.05)], 0.95)
            ]
        if len(aug_funcs) > 0:
            return t.Compose(aug_funcs)
        else:
            return None


if __name__ == '__main__':
    dst = Stanford3DDataset(
        data_root='/work/patrickwu2/S3DIS_processed',
        voxel_size=0.01,
    )
    print(len(dst))
    # print(dst[10])
    for i in range(len(dst)):
        x = dst[i]
        # print(x['coords'].min(), x['coords'].max())
