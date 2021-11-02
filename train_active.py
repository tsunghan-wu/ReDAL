#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm

# torch
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

# custom
from base_agent import BaseTrainer
from dataloader import get_dataset, get_active_dataset
from active_selection import get_active_selector
from utils.common import initialization, finalization


class Trainer(BaseTrainer):
    def __init__(self, args, logger):
        super().__init__(args, logger)

    def train(self, active_set):
        # prepare dataset
        train_dataset = active_set.label_dataset
        val_dataset = get_dataset(name=self.args.name, data_root=self.args.data_dir, imageset='val')
        self.sampler, self.train_dataset_loader = self.get_trainloader(train_dataset)
        self.val_sampler, self.val_dataset_loader = self.get_valloader(val_dataset)
        self.checkpoint_file = os.path.join(self.model_save_dir, f'checkpoint{active_set.selection_iter}.tar')

        # max epoch
        if active_set.selection_iter == 1:
            max_epoch = self.args.training_epoch
        else:
            max_epoch = self.args.finetune_epoch
        start_val_epoch = max_epoch - 20
        for epoch in tqdm(range(max_epoch)):
            validation = (epoch >= start_val_epoch)
            self.train_one_epoch(epoch, validation)


def main(rank, args):
    # initialization
    logger = initialization(rank, args)
    t_start = datetime.now()
    val_result = {}
    # Active Learning dataset
    active_set = get_active_dataset(args)
    active_selector = get_active_selector(args)

    # Active Learning iteration
    for selection_iter in range(1, args.max_iterations + 1):
        active_set.selection_iter = selection_iter

        if rank == 0:
            data_num = round(active_set.get_fraction_of_labeled_data() * 100)
            logger.info(f"AL {selection_iter}: Start Training ({data_num}% training data)")
        trainer = Trainer(args, logger)
        if selection_iter > 1:
            prevckpt_fname = os.path.join(args.model_save_dir, f'checkpoint{selection_iter-1}.tar')
            trainer.load_checkpoint(prevckpt_fname, rank)
        if args.distributed_training is True:
            dist.barrier()
        trainer.train(active_set)

        # load best checkpoint
        if args.distributed_training is True:
            dist.barrier()
        fname = os.path.join(args.model_save_dir, f'checkpoint{selection_iter}.tar')
        trainer.load_checkpoint(fname, rank)
        # evaluate the result
        val_return = trainer.validate(update_ckpt=False)
        if rank == 0:
            logger.info(f"AL {selection_iter}: Get best validation result")
            val_result[selection_iter] = val_return
        # active-select pool
        if rank == 0:
            logger.info(f"AL {selection_iter}: Select Next Batch")
        active_selector.select_next_batch(trainer, active_set, args.active_selection_size)
        if rank == 0:
            active_set.dump_datalist()
        if args.distributed_training is True:
            # Wait for rank 0 to write selection path
            dist.barrier()
            if rank != 0:
                active_set.load_datalist()
            dist.barrier()
    # finalization
    finalization(t_start, val_result, logger, args)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Active Learning framework.')

    # basic
    parser.add_argument('-n', '--name', choices=['s3dis', 'semantic_kitti', 'scannet'], default='s3dis',
                        help='training dataset (default: s3dis)')
    parser.add_argument('-d', '--data_dir', default='/tmp2/tsunghan/S3DIS_processed/')
    parser.add_argument('-p', '--model_save_dir', default='./test')
    parser.add_argument('-m', '--model', choices=['minkunet', 'spvcnn'], default='spvcnn',
                        help='training model (default: spvcnn)')

    # training
    parser.add_argument('--num_classes', type=int, default=13, help='number of classes in dataset')
    parser.add_argument('--ignore_idx', type=int, default=-100, help='ignore index')
    parser.add_argument('--training_epoch', type=int, default=200, help='initial training epoch')
    parser.add_argument('--finetune_epoch', type=int, default=50, help='finetune epoch')
    parser.add_argument('--train_batch_size', type=int, default=4, help='batch size for training (default: 4)')
    parser.add_argument('--val_batch_size', type=int, default=10, help='batch size for validation (default: 10)')
    parser.add_argument('--distributed_training', action='store_true', help='distributed training or not')
    parser.add_argument('--ddp_port', type=int, default=7122, help='DDP connection port')

    # Active Learning setting
    parser.add_argument('--max_iterations', type=int, default=10,
                        help='Number of active learning iterations (default: 10)')
    parser.add_argument('--active_method', type=str, required=True,
                        choices=['random', 'softmax_confidence', 'softmax_margin', 'softmax_entropy',
                                 'mc_dropout', 'coreset', 'segment_entropy'],
                        help='Active Learning Methods')
    parser.add_argument('--active_selection_size', type=int, default=4,
                        help='active selection size/images (default: 4)')

    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    if args.distributed_training is True:
        args.gpus = torch.cuda.device_count()
        mp.spawn(main, nprocs=args.gpus, args=(args,))
    else:
        main(0, args)
