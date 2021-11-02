import os
import json
import yaml
import numpy as np
from torch.utils import data
from torchsparse.utils import sparse_collate_fn, sparse_quantize
from torchsparse import SparseTensor


class RegionSemKITTI(data.Dataset):
    def __init__(self, data_path, voxel_size, imageset='train', init_lst=None):
        self.voxel_size = voxel_size
        with open("dataloader/semantic_kitti/semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']

        """
        -- SuperVoxel dict format --
        {
            "00_000000": [0, 1, 3, 4, .... K]
            "seq_id_scan_id": [supervoxel ids]
        }
        """

        self.imageset = imageset
        if imageset == 'active-label':
            with open('dataloader/semantic_kitti/init_data/init_label_large_region.json') as f:
                json_dict = json.load(f)
        elif imageset == 'active-ulabel':
            with open('dataloader/semantic_kitti/init_data/init_ulabel_large_region.json') as f:
                json_dict = json.load(f)
        else:
            raise Exception('Split must be train/val/test, or active-label/active-ulabel')

        self.im_idx = []
        self.supvox = {}
        if imageset in ['active-label', 'active-ulabel']:
            for k in json_dict:
                seq_id, scan_id = k.split('_')
                path = os.path.join(data_path, seq_id, 'velodyne', scan_id + '.bin')
                self.im_idx.append(path)
                self.supvox[path] = json_dict[k]
        self.angle = 0.0
        self.entropy_only = True

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        # Read Raw Lidar
        block_ = np.fromfile(self.im_idx[index], dtype=np.float32).reshape(-1, 4)
        block = np.zeros_like(block_)
        # Augmentation parameter
        if self.imageset == 'active-label':
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta),
                                 np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3]
        pc_ = np.round(block[:, :3] / self.voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        # Read supvox
        supvox_file = self.im_idx[index].replace('velodyne', 'supervoxel_large')
        supvox = np.fromfile(supvox_file, dtype=np.int32)
        # Read label
        if self.imageset == 'active-ulabel':
            all_labels = supvox
        elif self.imageset == 'active-label':
            label_file = self.im_idx[index].replace('velodyne', 'labels').replace('.bin', '.label')
            all_labels = np.fromfile(label_file, dtype=np.int32).reshape(-1)
            all_labels = all_labels & 0xFFFF
            all_labels = np.vectorize(self.learning_map.__getitem__)(all_labels).astype(np.uint8)
            all_labels -= 1  # 0 to 255 trick
            # Mask label
            preserving_labels = self.supvox[self.im_idx[index]]
            mask = np.isin(supvox, preserving_labels)
            all_labels = np.where(mask, all_labels, 255)

        labels_ = all_labels.astype(np.int64)  # instance labels

        feat_ = block

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
                'file_name': self.im_idx[index]
            }
        else:
            curvature_fn = self.im_idx[index].replace('velodyne', 'boundary').replace("bin", "npy")
            curvature = np.load(curvature_fn).astype(np.float32)
            colorgrad = np.zeros_like(curvature)
            return {
                'lidar': lidar,
                'targets': labels,
                'targets_mapped': labels_,
                'inverse_map': inverse_map,
                'file_name': self.im_idx[index],
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

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


if __name__ == "__main__":
    dataset = RegionSemKITTI("/work/patrickwu2/PCL_Seg_data/SemanticKitti/sequences", voxel_size=0.05)
    loader = data.DataLoader(dataset=dataset, batch_size=4, collate_fn=dataset.collate_fn,
                             shuffle=True, num_workers=4, pin_memory=True)

    for batch in loader:
        print(batch)
        exit()
