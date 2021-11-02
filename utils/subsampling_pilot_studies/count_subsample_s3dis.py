"""
Subsampling pilot studies (now useless)
"""

import os
import numpy as np

root_dir = '/tmp3/tsunghan/S3DIS_subsample'
train_split = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
id_to_name = {0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column', 5: 'window',
              6: 'door', 7: 'chair', 8: 'table', 9: 'bookcase', 10: 'sofa', 11: 'board', 12: 'clutter'}


for active_selection in range(1, 10):
    class_pts = [0 for _ in range(13)]
    selection_dir = os.path.join(root_dir, f'selection_{active_selection}')
    total_pts = 0
    items = 0
    for split in train_split:
        labels_dirname = os.path.join(selection_dir, split, 'labels')

        for fname in os.listdir(labels_dirname):
            labels_fname = os.path.join(labels_dirname, fname)
            labels = np.load(labels_fname)
            total_pts += labels.shape[0]
            values, cnts = np.unique(labels, return_counts=True)
            for value, cnt in zip(values, cnts):
                class_pts[value] += cnt
            items += 1
    print('-----------------------------')
    print(f'[selection {active_selection}]')
    print(f'Average: {total_pts // items} pts per scan')
    for i in range(13):
        print(f'{id_to_name[i]}: {class_pts[i]} pts, {(class_pts[i]/total_pts * 100):.5f}%')
