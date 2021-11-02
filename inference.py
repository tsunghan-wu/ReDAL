"""
visualize error on validation set (Area_5)
"""

# basic
import os
import sys
import random
import argparse
import numpy as np

# torch
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

# custom
from models import get_model
from active_selection.utils import SequentialDistributedSampler
from utils.s3dis_saver import S3DISSaver
from dataloader import get_dataset


class Tester:
    def __init__(self, args, active_iter, val_set, saver):
        self.args = args
        self.model_save_dir = args.model_save_dir
        self.active_iter = active_iter
        self.batch_size = args.batch_size
        self.distributed = args.distributed_training
        self.saver = saver
        if self.distributed is True:
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            pytorch_device = torch.device('cuda', local_rank)
            self.local_rank = local_rank
        else:
            pytorch_device = torch.device('cuda:0')
            self.local_rank = 0

        # prepare dataset
        self.dataset = val_set

        # prepare model
        self.NUM_CLASSES = self.dataset.NUM_CLASSES
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
        print("Class init done", flush=True)

    def inference(self):
        self.net.eval()
        if self.distributed is True:
            al_sampler = SequentialDistributedSampler(self.dataset, num_replicas=dist.get_world_size(),
                                                      rank=self.local_rank, batch_size=self.batch_size)
        else:
            al_sampler = None
        loader = torch.utils.data.DataLoader(dataset=self.dataset,
                                             batch_size=self.batch_size,
                                             collate_fn=self.dataset.collate_fn,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True, sampler=al_sampler)

        if self.distributed is True:
            idx = self.local_rank * al_sampler.num_samples
        else:
            idx = 0
        print(idx)

        with torch.no_grad():
            for i_iter_test, batch in enumerate(loader):
                # predict
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()
                inputs = batch['lidar']
                outputs = self.net(inputs)
                preds = outputs['final']
                invs = batch['inverse_map']
                all_labels = batch['targets_mapped']

                scene_pts = inputs.C.cpu().numpy()
                invsC = invs.C.cpu().numpy()
                invsF = invs.F.cpu().numpy()

                all_labels_C = all_labels.C.cpu().numpy()

                for batch_idx in range(self.batch_size):
                    fname = batch['file_name'][batch_idx]
                    assert fname == self.dataset.im_idx[idx]

                    cur_scene_pts = (scene_pts[:, -1] == batch_idx)
                    cur_inv = invsF[invsC[:, -1] == batch_idx]
                    output = preds[cur_scene_pts][cur_inv].argmax(1).cpu().numpy()

                    target = all_labels.F[all_labels_C[:, -1] == batch_idx].cpu().numpy()
                    feats = inputs.F[cur_scene_pts][cur_inv]
                    feats = feats.cpu().detach().numpy()

                    self.saver.save_pred_error(feats[:, 3:], output, target, fname, self.active_iter)
                    idx += 1
                    if idx >= len(self.dataset.im_idx):
                        break
                if idx >= len(self.dataset.im_idx):
                    break

    def load_checkpoint(self, fname, local_rank):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(fname, map_location=map_location)
        self.net.load_state_dict(checkpoint['model_state_dict'])


def main(rank, args):
    random.seed(1 + rank)
    np.random.seed(1 + rank)
    torch.manual_seed(7122)
    os.makedirs(args.output_dir, exist_ok=True)
    # Initialize DDP
    if args.distributed_training is True:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:7122', world_size=args.gpus, rank=rank)

    # Initialize saver
    if args.name == 's3dis':
        saver = S3DISSaver(args)
    if rank == 0:
        print('Save input / GT / Sup first.')
        # saver.save_input_gt_sup()
        print('Done.')

    # Active Learning iteration
    val_dataset = get_dataset(name=args.name, data_root=args.data_dir, imageset='val')
    for selection_iter in range(1, args.max_iterations + 1):

        tester = Tester(args, selection_iter, val_dataset, saver)
        # load best checkpoint
        fname = os.path.join(args.model_save_dir, f'checkpoint{selection_iter}.tar')
        tester.load_checkpoint(fname, rank)
        tester.inference()
        print(f"Finish Prediction {selection_iter}", flush=True)
        # save selection
        if args.distributed_training is True:
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
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for validation (default: 10)')
    parser.add_argument('--distributed_training', action='store_true', help='distributed training or not')

    # Active Learning setting
    parser.add_argument('--max_iterations', type=int, default=7,
                        help='Number of active learning iterations [default: 7]')

    args = parser.parse_args()
    args.output_dir = os.path.join(args.model_save_dir, 'val_result')
    print(' '.join(sys.argv))
    print(args)

    if args.distributed_training is True:
        args.gpus = torch.cuda.device_count()
        mp.spawn(main, nprocs=args.gpus, args=(args,))
    else:
        main(0, args)
