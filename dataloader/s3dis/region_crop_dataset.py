import os
import numpy as np

import dataloader.s3dis.transforms as t
from torchsparse.utils import sparse_collate_fn, sparse_quantize
from torchsparse import SparseTensor


class RegionCropStanford3DDataset:
    NUM_CLASSES = 13
    CLIP_BOUND = 4
    TEST_CLIP_BOUND = None
    IGNORE_LABEL = -100
    ROTATION_AXIS = 'z'

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = \
        ((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (-0.05, 0.05))

    def __init__(self, data_root, voxel_size, init_lst):
        # ASSIGN PARAMETERS
        self.voxel_size = voxel_size
        self.use_augs = {'scale': False, 'rotate': False, 'elastic': False, 'chromatic': False}
        assert init_lst is not None
        """
        -- SuperVoxel init_lst format --
        [
            (score (float), "path-name", supvox_id)
        ]
        """
        self.im_idx = []
        self.supvox = []
        for item in init_lst:
            _, path, supvox_id = item
            basename = '/'.join(path.split('/')[-3:])
            new_path = os.path.join(data_root, basename)
            self.im_idx.append(new_path)
            self.supvox.append(supvox_id)

        self.prevoxel_aug_func = self.build_prevoxel_aug_func()
        self.postvoxel_aug_func = self.build_postvoxel_aug_func()

    def __getitem__(self, idx):
        # Read data
        coords, feats, labels, supvox = self.load_data(self.im_idx[idx])

        # supervoxel cropping
        target_region = (supvox == self.supvox[idx])
        coords = coords[target_region]
        feats = feats[target_region]
        labels = labels[target_region]

        # Prevoxel Augmentation
        if self.prevoxel_aug_func is not None:
            coords, feats, labels = self.prevoxel_aug_func(coords, feats, labels)

        # Voxelize
        pc_ = np.round(coords / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        # Postvoxel transformation
        if self.postvoxel_aug_func is not None:
            pc_, feats, labels = self.postvoxel_aug_func(pc_, feats, labels)

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
        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.im_idx[idx]
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)

    def load_data(self, fn):
        coords_fn = fn
        feats_fn = fn.replace('coords', 'rgb')
        labels_fn = fn.replace('coords', 'labels')
        supvox_fn = fn.replace('coords', 'supervoxel')
        coords = np.load(coords_fn).astype(np.float32)
        feats = np.load(feats_fn).astype(np.float32)
        labels = np.load(labels_fn).astype(np.int32)
        supvox = np.load(supvox_fn).astype(np.int32)
        return coords, feats, labels, supvox

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
