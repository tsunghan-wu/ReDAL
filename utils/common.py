import os
import sys
import torch
import random
import logging
import numpy as np
from datetime import datetime
import torch.distributed as dist
from dateutil.relativedelta import relativedelta


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


def timediff(t_start, t_end):
    t_diff = relativedelta(t_end, t_start)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


def initialization(rank, args):
    # set random seed
    random.seed(1 + rank)
    np.random.seed(1 + rank)
    torch.manual_seed(7122)
    # Initialize Logging
    if rank == 0:
        logger = initialize_logging(args.model_save_dir)
        logger.info(' '.join(sys.argv))
        logger.info(args)
    else:
        logger = None
    # Initialize DDP
    if args.distributed_training is True:
        dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.ddp_port}',
                                world_size=args.gpus, rank=rank)
    return logger


def finalization(rank, t_start, val_result, logger, args):
    if rank == 0:
        # End Experiment
        t_end = datetime.now()
        logger.info(f"{'%'*20} Experiment Report {'%'*20}")
        logging.info(f"0. AL Methods: {args.active_method}")
        logging.info(f"1. Takes: {timediff(t_start, t_end)}")
        logging.info(f"2. Log dir: {args.model_save_dir} (with selection json & model checkpoint)")
        logging.info("3. Validation mIoU (Be sure to submit to google form)")
        for selection_iter in range(1, args.max_iterations + 1):
            logging.info(f"AL {selection_iter}: {val_result[selection_iter]}")
        logger.info(f"{'%'*20} Experiment End {'%'*20}")
