# torch
import torch
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler

# model, dataset, utils
from models import get_model
from utils.miou import MeanIoU


class BaseTrainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.model_save_dir = args.model_save_dir
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

        # prepare model
        self.num_classes = args.num_classes

        self.net = get_model(name=args.name, model=args.model, num_classes=self.num_classes)

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
        # Optimizer
        self.optimizer = optim.Adam(self.net.parameters())
        self.loss_fun = torch.nn.CrossEntropyLoss(ignore_index=self.args.ignore_idx)
        print("Class init done", flush=True)

    def get_trainloader(self, dataset):
        if self.distributed is True:
            sampler = DistributedSampler(dataset, num_replicas=self.args.gpus, rank=self.local_rank)
        else:
            sampler = None
        dataset_loader = \
            torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.train_batch_size,
                                        collate_fn=dataset.collate_fn,
                                        sampler=sampler, shuffle=(sampler is None),
                                        num_workers=4, pin_memory=True)
        return sampler, dataset_loader

    def get_valloader(self, dataset):
        if self.distributed is True:
            sampler = DistributedSampler(dataset, num_replicas=self.args.gpus, rank=self.local_rank, shuffle=False)
        else:
            sampler = None

        dataset_loader = \
            torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.val_batch_size,
                                        collate_fn=dataset.collate_fn, sampler=sampler, shuffle=False,
                                        num_workers=4, pin_memory=True)
        return sampler, dataset_loader

    def train(self):
        raise NotImplementedError

    def train_one_epoch(self, epoch, validation):
        self.net.train()
        if self.local_rank == 0:
            self.logger.info('**** EPOCH %03d ****' % (epoch))
        if self.distributed is True:
            self.sampler.set_epoch(epoch)
        for i_iter, batch in enumerate(self.train_dataset_loader):
            # training
            for key, value in batch.items():
                if 'name' not in key:
                    batch[key] = value.cuda()
            inputs = batch['lidar']
            targets = batch['targets'].F.long().cuda(non_blocking=True)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward + backward + optimize
            torch.cuda.synchronize()
            outputs = self.net(inputs)
            preds = outputs['final']

            loss = self.loss_fun(preds, targets)
            loss.backward()
            self.optimizer.step()
        if validation is True:
            if self.local_rank == 0:
                self.logger.info('**** EVAL EPOCH %03d ****' % (epoch))
            self.validate()

    def validate(self, update_ckpt=True):
        self.net.eval()
        iou_helper = MeanIoU(self.num_classes, self.args.ignore_idx, distributed=self.distributed)
        iou_helper._before_epoch()

        with torch.no_grad():
            for i_iter_val, batch in enumerate(self.val_dataset_loader):
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()

                inputs = batch['lidar']
                targets = batch['targets'].F.long().cuda(non_blocking=True)

                outputs = self.net(inputs)
                preds = outputs['final']

                invs = batch['inverse_map']
                all_labels = batch['targets_mapped']

                _outputs = []
                _targets = []

                for idx in range(invs.C[:, -1].max()+1):
                    cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
                    cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                    cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                    outputs_mapped = preds[cur_scene_pts][
                        cur_inv].argmax(1)
                    targets_mapped = all_labels.F[cur_label]
                    _outputs.append(outputs_mapped)
                    _targets.append(targets_mapped)

                outputs = torch.cat(_outputs, 0)
                targets = torch.cat(_targets, 0)

                output_dict = {
                    'outputs': outputs,
                    'targets': targets
                }
                iou_helper._after_step(output_dict)
            val_miou, ious = iou_helper._after_epoch()
            # Prepare Logging
            iou_table = []
            iou_table.append(f'{val_miou:.2f}')
            for class_iou in ious:
                iou_table.append(f'{class_iou:.2f}')
            iou_table_str = ','.join(iou_table)
            # save model if performance is improved
            if update_ckpt is False:
                return iou_table_str

            if self.local_rank == 0:
                self.logger.info('[Validation Result]')
                self.logger.info('%s' % (iou_table_str))
                if self.best_iou < val_miou:
                    self.best_iou = val_miou
                    checkpoint = {
                                    'model_state_dict': self.net.state_dict(),
                                    'opt_state_dict': self.optimizer.state_dict()
                                 }
                    torch.save(checkpoint, self.checkpoint_file)

                self.logger.info('Current val miou is %.3f %%, while the best val miou is %.3f %%'
                                 % (val_miou, self.best_iou))
            return iou_table_str

    def load_checkpoint(self, fname, local_rank):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(fname, map_location=map_location)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['opt_state_dict'])
