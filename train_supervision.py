#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta

# torch
import torch
import torch.multiprocessing as mp

# model, dataset, utils
from base_agent import BaseTrainer
from dataloader import get_dataset
from utils.common import initialization


class Trainer(BaseTrainer):
    def __init__(self, args, logger):
        super().__init__(args, logger)

    def train(self):
        # prepare dataset
        train_dataset = get_dataset(name=self.args.name, data_root=self.args.data_dir, imageset='train')
        val_dataset = get_dataset(name=self.args.name, data_root=self.args.data_dir, imageset='val')
        self.sampler, self.train_dataset_loader = self.get_trainloader(train_dataset)
        self.val_sampler, self.val_dataset_loader = self.get_valloader(val_dataset)
        self.checkpoint_file = os.path.join(self.model_save_dir, 'checkpoint.tar')

        # max epoch
        max_epoch = self.args.training_epoch
        start_val_epoch = max_epoch - 20
        for epoch in tqdm(range(max_epoch)):
            validation = (epoch >= start_val_epoch)
            self.train_one_epoch(epoch, validation)


def initialize_logging(exp_dir):
    # mkdir
    os.makedirs(exp_dir, exist_ok=True)
    log_fname = os.path.join(exp_dir, 'log_train.txt')
    LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    DATE_FORMAT = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
    logger = logging.getLogger("Trainer")
    logger.info(f"{'-'*20} New Experiment {'-'*20}")
    return logger


def timediff(t_start, t_end):
    t_diff = relativedelta(t_end, t_start)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


def main(rank, args):
    # initialization
    logger = initialization(rank, args)
    t_start = datetime.now()
    trainer = Trainer(args, logger)
    trainer.train()
    # Evaluate on validation set
    fname = os.path.join(args.model_save_dir, 'checkpoint.tar')
    trainer.load_checkpoint(fname, rank)
    result = trainer.validate(update_ckpt=False)
    t_end = datetime.now()
    if rank == 0:
        # End Experiment
        t_end = datetime.now()
        logger.info(f"{'%'*20} Experiment Report {'%'*20}")
        logger.info("0. Methods: Fully Supervision")
        logger.info(f"1. Takes: {timediff(t_start, t_end)}")
        logger.info(f"2. Log dir: {args.model_save_dir} (with selection json & model checkpoint)")
        logger.info("3. Validation mIoU (Be sure to submit to google form)")
        logger.info(result)
        logger.info(f"{'%'*20} Experiment End {'%'*20}")


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    # basic
    parser.add_argument('-n', '--name', choices=['s3dis', 'semantic_kitti', 'scannet'], default='s3dis',
                        help='training dataset (default: s3dis)')
    parser.add_argument('-d', '--data_dir', default='/tmp2/tsunghan/S3DIS_processed/')
    parser.add_argument('-p', '--model_save_dir', default='./test')
    parser.add_argument('-m', '--model', choices=['minkunet', 'spvcnn'], default='spvcnn',
                        help='training model (default: spvcnn)')
    # training related
    parser.add_argument('--num_classes', type=int, default=13, help='number of classes in dataset')
    parser.add_argument('--ignore_idx', type=int, default=-100, help='ignore index')
    parser.add_argument('--training_epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--train_batch_size', type=int, default=4, help='batch size for training (default: 4)')
    parser.add_argument('--val_batch_size', type=int, default=10, help='batch size for validation (default: 10)')
    parser.add_argument('--distributed_training', action='store_true', help='distributed training or not')
    parser.add_argument('--ddp_port', type=int, default=7122, help='DDP connection port')

    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    if args.distributed_training is True:
        args.gpus = torch.cuda.device_count()
        mp.spawn(main, nprocs=args.gpus, args=(args,))
    else:
        main(0, args)
