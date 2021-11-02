#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import sys
import argparse
import numpy as np
from dataloader import get_active_dataset


def analysis_label(dataset_name, label_dataset):
    im_idx = label_dataset.im_idx
    supvox_dict = label_dataset.supvox
    total_pts = 0
    if dataset_name == 's3dis':
        analysis_data = {0: ['ceiling', 0], 1: ['floor', 0]}
        for fname in im_idx:
            # load data
            supvox_fn = fname.replace('coords', 'supervoxel')
            supvox = np.load(supvox_fn).reshape(-1)
            labels_fn = fname.replace('coords', 'labels')
            labels = np.load(labels_fn).reshape(-1)
            # effective labeled data
            mask = np.isin(supvox, supvox_dict[fname])
            effective_labels = np.where(mask, labels, -1)
            total_pts += mask.sum()
            for key in analysis_data:
                num = (effective_labels == key).sum()
                analysis_data[key][1] += num
        for key in analysis_data:
            class_name = analysis_data[key][0]
            class_num_pts = analysis_data[key][1]
            data_percent = round((class_num_pts / total_pts) * 100, 1)
            print(f'  - {class_name}: {data_percent}%')


def main(args):

    # Active Learning dataset
    active_set = get_active_dataset(args, mode='region')

    # Active Learning iteration
    print(f"[Active Strategy: {args.active_method}]")
    for selection_iter in range(1, args.max_iterations + 1):
        print(f"--- AL {selection_iter} ---")
        active_set.selection_iter = selection_iter

        # analyze number of selected region
        data_num = round(active_set.get_fraction_of_labeled_data() * 100, 1)
        region_num = active_set.get_number_of_labeled_region()
        print(f"- {data_num}% labeled points / {region_num} labeled region")
        # analyze selected label
        analysis_label(args.name, active_set.label_dataset)

        # update active-selected data
        active_set.load_datalist(convert_root=True)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')

    # basic
    parser.add_argument('-n', '--name', choices=['s3dis', 'semantic_kitti'], default='s3dis',
                        help='training dataset (default: s3dis)')
    parser.add_argument('-d', '--data_dir', default='/tmp2/tsunghan/S3DIS_processed/')
    parser.add_argument('-p', '--model_save_dir', default='./test')

    # Active Learning setting
    parser.add_argument('--max_iterations', type=int, default=10,
                        help='Number of active learning iterations [default: 8]')
    parser.add_argument('--active_method', type=str, required=True,
                        choices=['random', 'softmax_confidence', 'softmax_margin', 'softmax_entropy',
                                 'mc_dropout', 'coreset', 'spatial_entropy', 'representative', 'ssm'],
                        help='Active Learning Methods')
    parser.add_argument('--active_percent', type=float, default=2.0, help='active selection percent')

    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
