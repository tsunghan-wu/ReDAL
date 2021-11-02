from typing import Any, Dict
import numpy as np
import torch
import torch.distributed as dist


class MeanIoU():
    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou',
                 distributed: bool = True) -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.distributed = distributed

    def _before_epoch(self) -> None:
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]
        if type(outputs) != np.ndarray:
            for i in range(self.num_classes):
                self.total_seen[i] += torch.sum(targets == i).item()
                self.total_correct[i] += torch.sum(
                    (targets == i) & (outputs == targets)).item()
                self.total_positive[i] += torch.sum(
                    outputs == i).item()
        else:
            for i in range(self.num_classes):
                self.total_seen[i] += np.sum(targets == i)
                self.total_correct[i] += np.sum((targets == i)
                                                & (outputs == targets))
                self.total_positive[i] += np.sum(outputs == i)

    def _after_epoch(self) -> None:
        if self.distributed is True:
            for i in range(self.num_classes):
                tmp_total_seen = torch.tensor(self.total_seen[i]).cuda()
                dist.all_reduce(tmp_total_seen, op=dist.ReduceOp.SUM)
                self.total_seen[i] = tmp_total_seen.item()

                tmp_total_correct = torch.tensor(self.total_correct[i]).cuda()
                dist.all_reduce(tmp_total_correct, op=dist.ReduceOp.SUM)
                self.total_correct[i] = tmp_total_correct.item()

                tmp_total_positive = torch.tensor(self.total_positive[i]).cuda()
                dist.all_reduce(tmp_total_positive, op=dist.ReduceOp.SUM)
                self.total_positive[i] = tmp_total_positive.item()

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i] + self.total_positive[i] - self.total_correct[i])
                ious.append(cur_iou)

        miou = np.mean(ious)
        # 0.xx to 100%
        miou = miou * 100
        ious = [num * 100 for num in ious]
        return miou, ious
