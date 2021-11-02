"""
Analysis nuber of labels in each class (or percentages) on S3DIS dataset.
"""

import os
import numpy as np

root_dir = '/tmp3/tsunghan/S3DIS_processed'
train_split = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
total_pts = 0
items = 0
id_to_name = {0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column', 5: 'window',
              6: 'door', 7: 'chair', 8: 'table', 9: 'bookcase', 10: 'sofa', 11: 'board', 12: 'clutter'}

class_pts = [0 for _ in range(13)]

for split in train_split:
    labels_dirname = os.path.join(root_dir, split, 'labels')

    for fname in os.listdir(labels_dirname):
        labels_fname = os.path.join(labels_dirname, fname)
        labels = np.load(labels_fname)
        total_pts += labels.shape[0]
        values, cnts = np.unique(labels, return_counts=True)
        for value, cnt in zip(values, cnts):
            class_pts[value] += cnt
        items += 1
print(total_pts//items)
for i in range(13):
    print(f'{id_to_name[i]}: {class_pts[i]} pts, {(class_pts[i]/total_pts * 100):.5f}%')
