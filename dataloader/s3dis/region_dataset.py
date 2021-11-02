import os
import json
import numpy as np
import dataloader.s3dis.transforms as t
from torchsparse.utils import sparse_collate_fn, sparse_quantize
from torchsparse import SparseTensor


class RegionStanford3DDataset:
    NUM_CLASSES = 13
    CLIP_BOUND = 4
    TEST_CLIP_BOUND = None
    IGNORE_LABEL = -100
    ROTATION_AXIS = 'z'

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = \
        ((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (-0.05, 0.05))

    def __init__(self, data_root, voxel_size, imageset='train', init_lst=None):
        # ASSIGN PARAMETERS
        self.data_root = data_root
        self.voxel_size = voxel_size

        """
        -- SuperVoxel dict format --
        {
            "00_000000": [0, 1, 3, 4, .... K]
            "seq_id_scan_id": [supervoxel ids]
        }
        """

        self.imageset = imageset
        if imageset == 'active-label':
            with open('dataloader/s3dis/init_data/init_label_region.json') as f:
                json_dict = json.load(f)
            self.use_augs = {'scale': True, 'rotate': True, 'elastic': True, 'chromatic': True}
        elif imageset == 'active-ulabel':
            with open('dataloader/s3dis/init_data/init_ulabel_region.json') as f:
                json_dict = json.load(f)
            self.use_augs = {'scale': False, 'rotate': False, 'elastic': False, 'chromatic': False}

        if self.imageset not in ['active-label']:
            for key, val in self.use_augs.items():
                assert val is False, f"{key} should be False during evaluation"

        self.im_idx = []
        self.supvox = {}
        if imageset in ['active-label', 'active-ulabel']:
            for k in json_dict:
                seq_id, scan_id = k.split('#')
                path = os.path.join(self.data_root, seq_id, 'coords', scan_id + '.npy')
                self.im_idx.append(path)
                self.supvox[path] = json_dict[k]
        self.im_idx.sort()
        self.prevoxel_aug_func = self.build_prevoxel_aug_func()
        self.postvoxel_aug_func = self.build_postvoxel_aug_func()
        self.force_label = False
        self.entropy_only = True

    def set_force_label(self, flag):
        assert flag in [True, False]
        self.force_label = flag

    def __getitem__(self, idx):
        # Read data
        coords, feats, labels, supvox = self.load_data(self.im_idx[idx])
        # Prevoxel Augmentation
        if self.prevoxel_aug_func is not None:
            coords, feats, labels = self.prevoxel_aug_func(coords, feats, labels)

        # Voxelize
        pc_ = np.round(coords / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        # Postvoxel transformation
        if self.postvoxel_aug_func is not None:
            pc_, feats, labels = self.postvoxel_aug_func(pc_, feats, labels)

        if self.imageset == 'active-ulabel':
            if self.force_label is False:  # Normal
                labels = supvox.reshape(-1)
            else:  # Visualization
                labels = labels.reshape(-1)
        elif self.imageset == 'active-label':
            labels = labels.reshape(-1)
            preserving_labels = self.supvox[self.im_idx[idx]]
            mask = np.isin(supvox, preserving_labels)
            # mask = nb_isin(supvox, np.array(preserving_labels).astype(np.int32))
            labels = np.where(mask, labels, -100)
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
        if self.entropy_only is True:
            return {
                'lidar': lidar,
                'targets': labels,
                'targets_mapped': labels_,
                'inverse_map': inverse_map,
                'file_name': self.im_idx[idx]
            }
        else:
            curvature_fn = self.im_idx[idx].replace('coords', 'boundary')
            colorgrad_fn = self.im_idx[idx].replace('coords', 'colorgrad')
            curvature = np.load(curvature_fn).astype(np.float32)
            colorgrad = np.load(colorgrad_fn).astype(np.float32)
            return {
                'lidar': lidar,
                'targets': labels,
                'targets_mapped': labels_,
                'inverse_map': inverse_map,
                'file_name': self.im_idx[idx],
                'curvature': curvature,
                'colorgrad': colorgrad
            }

    def collate_fn(self, inputs):
        if self.entropy_only is True:
            return sparse_collate_fn(inputs)
        else:
            sparse_key = ['lidar', 'targets', 'targets_mapped', 'inverse_map', 'file_name']
            dense_key = ['curvature', 'colorgrad']
            N = len(inputs)
            sparse_batch = [dict() for _ in range(N)]
            for i in range(N):
                for key in sparse_key:
                    sparse_batch[i][key] = inputs[i][key]
            output_batch = sparse_collate_fn(sparse_batch)
            for key in dense_key:
                output_batch[key] = [one_batch[key] for one_batch in inputs]
            return output_batch

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
