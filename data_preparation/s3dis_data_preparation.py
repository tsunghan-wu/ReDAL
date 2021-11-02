# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import numpy as np
import os

# Dataset Path (Need to modify yourself)
STANFORD_3D_IN_PATH = './Stanford3dDataset_v1.2_Aligned_Version'
STANFORD_3D_OUT_PATH = './S3DIS_processed'

# Error line in Stanford3dDataset_v1.2_Aligned_Version/Area_5/hallway_6/Annotations/ceiling_1.txt
# could not convert string to float: '185\x00187'


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Stanford3DDatasetConverter:
    name_to_labels = {'ceiling': 0,
                      'floor': 1,
                      'wall': 2,
                      'beam': 3,
                      'column': 4,
                      'window': 5,
                      'door': 6,
                      'chair': 7,
                      'table': 8,
                      'bookcase': 9,
                      'sofa': 10,
                      'board': 11,
                      'clutter': 12,
                      'stairs': 12}     # Stairs --> clutter (follow KPConv)

    @classmethod
    def read_txt(cls, txtfile):
        # Read txt file and parse its content.
        with open(txtfile) as f:
            pointcloud = []
            for line in f:
                try:
                    data = [float(li) for li in line.split()]
                    if len(data) < 3:
                        # Prevent empty line in some file
                        raise Exception("Line with less than 3 digits")
                    pointcloud += [data]
                except Exception as e:
                    print(e, txtfile, flush=True)
                    continue

        # Load point cloud to named numpy array.
        pointcloud = np.array(pointcloud).astype(np.float32)
        assert pointcloud.shape[1] == 6
        xyz = pointcloud[:, :3].astype(np.float32)
        rgb = pointcloud[:, 3:].astype(np.uint8)
        return xyz, rgb

    @classmethod
    def convert_to_npy(cls, root_path, out_path):
        """Convert Stanford3DDataset to PLY format that is compatible with
        Synthia dataset. Assumes file structure as given by the dataset.
        Outputs the processed PLY files to `STANFORD_3D_OUT_PATH`.
        """

        txtfiles = glob.glob(os.path.join(root_path, '*/*/*.txt'))
        for idx, txtfile in enumerate(txtfiles):
            print(f"{idx+1} / {len(txtfiles)}", flush=True)
            file_sp = os.path.normpath(txtfile).split(os.path.sep)
            target_path = os.path.join(out_path, file_sp[-3])
            # Output Filename
            out_coords = os.path.join(target_path, "coords", file_sp[-2] + '.npy')
            out_rgb = os.path.join(target_path, "rgb", file_sp[-2] + '.npy')
            out_labels = os.path.join(target_path, "labels", file_sp[-2] + '.npy')

            os.makedirs(os.path.join(target_path, "coords"), exist_ok=True)
            os.makedirs(os.path.join(target_path, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(target_path, "labels"), exist_ok=True)
            if os.path.exists(out_coords):
                print(out_coords, ' exists')
                continue

            annotation, _ = os.path.split(txtfile)
            subclouds = glob.glob(os.path.join(annotation, 'Annotations/*.txt'))
            coords, feats, labels = [], [], []
            for inst, subcloud in enumerate(subclouds):
                # Read ply file and parse its rgb values.
                xyz, rgb = cls.read_txt(subcloud)
                _, annotation_subfile = os.path.split(subcloud)
                clsidx = cls.name_to_labels[annotation_subfile.split('_')[0]]

                coords.append(xyz)
                feats.append(rgb)
                labels.append(np.ones((len(xyz), 1), dtype=np.int32) * clsidx)
            if len(coords) == 0:
                print(txtfile, ' has 0 files.')
            else:
                # Concat
                coords = np.concatenate(coords, 0)
                feats = np.concatenate(feats, 0)
                labels = np.concatenate(labels, 0)
                np.save(out_coords, coords)
                np.save(out_rgb, feats)
                np.save(out_labels, labels)


if __name__ == '__main__':
    Stanford3DDatasetConverter.convert_to_npy(STANFORD_3D_IN_PATH, STANFORD_3D_OUT_PATH)
