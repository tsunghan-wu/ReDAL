#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import os
import sys
import random
import pickle
import logging
import argparse
import numpy as np

# torch
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from active_selection.dist_sampler import SequentialDistributedSampler

# custom
from models import get_model
from dataloader import get_dataset


class Trainer:
    def __init__(self, args, active_iter, pool_dataset, logger):
        self.args = args
        self.logger = logger
        self.model_save_dir = args.model_save_dir
        self.active_iter = active_iter
        self.best_iou = 0
        self.distributed = args.distributed_training
        if self.distributed is True:
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            pytorch_device = torch.device('cuda', local_rank)
            self.local_rank = local_rank
        else:
            pytorch_device = torch.device('cuda:0')
            self.local_rank = 0

        # prepare dataset
        self.pool_dataset = pool_dataset

        # prepare model
        self.NUM_CLASSES = pool_dataset.NUM_CLASSES
        self.net = get_model(name=args.name, model=args.model, num_classes=self.NUM_CLASSES)

        if self.distributed is True:
            # self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            self.net.to(pytorch_device)
            self.net = \
                torch.nn.parallel.DistributedDataParallel(self.net,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
        else:
            self.net.to(pytorch_device)

        # Sampler
        if args.distributed_training is True:
            self.sampler = SequentialDistributedSampler(pool_dataset, num_replicas=args.gpus,
                                                        rank=local_rank, batch_size=args.val_batch_size)
        else:
            self.sampler = None

        # Loader
        self.pool_dataset_loader = \
            torch.utils.data.DataLoader(dataset=pool_dataset, batch_size=args.val_batch_size,
                                        collate_fn=pool_dataset.collate_fn,
                                        sampler=self.sampler, shuffle=(self.sampler is None),
                                        num_workers=4, pin_memory=True)

        print("Class init done", flush=True)

    def extract_region_feature(self, update_ckpt=True):
        # validation
        self.net.eval()
        if self.distributed is True:
            idx = self.local_rank * self.sampler.num_samples
        else:
            idx = 0
        print(idx)

        feature = []

        with torch.no_grad():
            for i_iter_test, batch in enumerate(self.pool_dataset_loader):
                # predict
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()
                inputs = batch['lidar']
                outputs = self.net(inputs)
                feats = outputs['feat']
                featC = feats.C.cpu().numpy()
                featF = feats.F.cpu().numpy()

                for batch_idx in range(self.args.val_batch_size):
                    fname = batch['file_name'][batch_idx]
                    assert fname == self.pool_dataset.im_idx[idx]

                    feat = featF[featC[:, -1] == batch_idx].mean(axis=0).reshape(1, -1)
                    feature.append(feat)

                    idx += 1
                    if idx >= len(self.pool_dataset.im_idx):
                        break
                if idx >= len(self.pool_dataset.im_idx):
                    break
        feat_np = np.concatenate(feature, 0)
        region_feature_dir = os.path.join(self.model_save_dir, "Region_Feature")
        os.makedirs(region_feature_dir, exist_ok=True)
        fname = os.path.join(region_feature_dir, f"region_feature_AL{self.active_iter}_rank{self.local_rank}.npy")
        np.save(fname, feat_np)

    def load_checkpoint(self, fname, local_rank):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(fname, map_location=map_location)
        self.net.load_state_dict(checkpoint)


def initialize_logging(exp_dir):
    # mkdir
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "AL_record"), exist_ok=True)
    log_fname = os.path.join(exp_dir, 'log_train.txt')
    LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    DATE_FORMAT = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
    logger = logging.getLogger("Trainer")
    logger.info(f"{'-'*20} New Experiment {'-'*20}")
    return logger


def main(rank, args):
    random.seed(1 + rank)
    np.random.seed(1 + rank)
    torch.manual_seed(7122)
    # Initialize Logging
    if rank == 0:
        logger = initialize_logging(args.model_save_dir)
        logger.info(args)
    else:
        logger = None
    # Initialize DDP
    if args.distributed_training is True:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:7122', world_size=args.gpus, rank=rank)

    # Active Learning iteration
    for selection_iter in range(1, args.max_iterations + 1):
        # Load selection
        selection_path = os.path.join(args.model_save_dir, f'selection_{selection_iter:02d}.pkl')
        with open(selection_path, 'rb') as f:
            selection_path = pickle.load(f)
        # region-crop dataset
        pool_dataset = get_dataset(name=args.name, data_root=args.data_dir,
                                   imageset='region-crop', init_lst=selection_path)
        trainer = Trainer(args, selection_iter, pool_dataset, logger)

        # load best checkpoint
        if args.distributed_training is True:
            dist.barrier()
        fname = os.path.join(args.model_save_dir, f'checkpoint{selection_iter}.tar')
        trainer.load_checkpoint(fname, rank)
        # evaluate the result
        trainer.extract_region_feature(update_ckpt=False)
        if args.distributed_training is True:
            # Wait for rank 0 to write selection path
            dist.barrier()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')

    # basic
    parser.add_argument('-n', '--name', choices=['s3dis', 'semantic_kitti'], default='s3dis',
                        help='training dataset (default: s3dis)')
    parser.add_argument('-d', '--data_dir', default='/tmp2/tsunghan/S3DIS_processed/')
    parser.add_argument('-p', '--model_save_dir', default='./test')
    parser.add_argument('-m', '--model', choices=['minkunet', 'spvcnn'], default='spvcnn',
                        help='training model (default: spvcnn)')

    # training
    parser.add_argument('--ignore_idx', type=int, default=-100, help='ignore index')
    parser.add_argument('--max_epoch', type=int, default=100, help='maximum epoch')
    parser.add_argument('--train_batch_size', type=int, default=4, help='batch size for training (default: 4)')
    parser.add_argument('--val_batch_size', type=int, default=10, help='batch size for validation (default: 10)')
    parser.add_argument('--distributed_training', action='store_true', help='distributed training or not')

    # Active Learning setting
    parser.add_argument('--max_iterations', type=int, default=10,
                        help='Number of active learning iterations [default: 10]')
    parser.add_argument('--active_method', type=str, required=True,
                        choices=['random', 'softmax_confidence', 'softmax_margin', 'softmax_entropy',
                                 'mc_dropout', 'coreset', 'segment_entropy'],
                        help='Active Learning Methods')
    parser.add_argument('--active_selection_size', type=int, default=4, help='active selection size')

    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    if args.distributed_training is True:
        args.gpus = torch.cuda.device_count()
        mp.spawn(main, nprocs=args.gpus, args=(args,))
    else:
        main(0, args)
