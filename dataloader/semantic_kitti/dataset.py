import os
import json
import yaml
import numpy as np
from torch.utils import data
from torchsparse.utils import sparse_collate_fn, sparse_quantize
from torchsparse import SparseTensor


class SemKITTI(data.Dataset):
    NUM_CLASSES = 19

    def __init__(self, data_path, voxel_size, imageset='train', init_lst=None):
        self.voxel_size = voxel_size
        with open("dataloader/semantic_kitti/semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']

        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        elif imageset == 'active-label':
            with open('dataloader/semantic_kitti/init_data/init_label_scan.json') as f:
                lst = json.load(f)
        elif imageset == 'active-ulabel':
            with open('dataloader/semantic_kitti/init_data/init_ulabel_scan.json') as f:
                lst = json.load(f)
        elif imageset == 'custom-set':
            lst = init_lst
        else:
            raise Exception('Split must be train/val/test, or active-label/active-ulabel')

        self.im_idx = []
        if imageset in ['train', 'val', 'test']:
            for i_folder in split:
                self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))
        elif imageset in ['active-label', 'active-ulabel']:
            self.im_idx = [os.path.join(data_path, i) for i in lst]
        elif imageset == 'custom-set':
            self.im_idx = lst
        self.angle = 0.0
        self.return_supvox = False

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        # Read Raw Lidar
        block_ = np.fromfile(self.im_idx[index], dtype=np.float32).reshape(-1, 4)
        block = np.zeros_like(block_)
        # Augmentation parameter
        if self.imageset in ['train', 'active-label']:
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

        # Read label
        if self.imageset == 'test':
            all_labels = np.zeros_like(pc_[:, 0], dtype=np.int32)
        else:
            if self.return_supvox is False:
                label_file = self.im_idx[index].replace('velodyne', 'labels').replace('.bin', '.label')
                all_labels = np.fromfile(label_file, dtype=np.int32).reshape(-1)
                all_labels = all_labels & 0xFFFF
                all_labels = np.vectorize(self.learning_map.__getitem__)(all_labels).astype(np.uint8)
                all_labels -= 1  # 0 to 255 trick
            elif self.return_supvox is True:
                supvox_file = self.im_idx[index].replace('velodyne', 'supervoxel')
                supvoxs = np.fromfile(supvox_file, dtype=np.int32).reshape(-1)
                all_labels = supvoxs
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

        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.im_idx[index]
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)

    def label_to_supvox(self):
        self.return_supvox = True

    def supvox_to_label(self):
        self.return_supvox = False


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


if __name__ == "__main__":
    dataset = SemKITTI("/work/patrickwu2/PCL_Seg_data/SemanticKitti/sequences", voxel_size=0.05)
    loader = data.DataLoader(dataset=dataset, batch_size=4, collate_fn=dataset.collate_fn,
                             shuffle=True, num_workers=4, pin_memory=True)

    for batch in loader:
        print(batch)
        exit()
