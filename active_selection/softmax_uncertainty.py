import os
import json
import pandas as pd
import torch
from tqdm import tqdm
import torch.distributed as dist
from active_selection.utils import get_al_loader


def softmax_confidence(preds):
    prob = torch.nn.functional.softmax(preds, dim=1)
    CONF = torch.max(prob, 1)[0]
    CONF *= -1  # The small the better --> Reverse it makes it the large the better
    return CONF


def softmax_margin(preds):
    prob = torch.nn.functional.softmax(preds, dim=1)
    TOP2 = torch.topk(prob, 2, dim=1)[0]
    MARGIN = TOP2[:, 0] - TOP2[:, 1]
    MARGIN *= -1   # The small the better --> Reverse it makes it the large the better
    return MARGIN


def softmax_entropy(preds):
    # Softmax Entropy
    prob = torch.nn.functional.softmax(preds, dim=1)
    ENT = torch.mean(-prob * torch.log2(prob + 1e-12), dim=1)  # The large the better
    return ENT


class SoftmaxUncertaintySelector:

    def __init__(self, batch_size, num_workers, active_method):
        self.batch_size = batch_size
        self.num_workers = num_workers
        assert active_method in ['softmax_confidence', 'softmax_margin', 'softmax_entropy']
        if active_method == 'softmax_confidence':
            self.uncertain_handler = softmax_confidence
        if active_method == 'softmax_margin':
            self.uncertain_handler = softmax_margin
        if active_method == 'softmax_entropy':
            self.uncertain_handler = softmax_entropy

    def calculate_scores(self, trainer, pool_set):
        model = trainer.net
        model.eval()
        loader, idx = get_al_loader(trainer, pool_set, self.batch_size, self.num_workers)
        print(idx)

        scores = []
        tqdm_loader = tqdm(loader, total=len(loader))
        with torch.no_grad():
            for i_iter_test, batch in enumerate(tqdm_loader):
                # predict
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()

                inputs = batch['lidar']
                outputs = model(inputs)
                preds = outputs['final']
                invs = batch['inverse_map']

                scene_pts = inputs.C.cpu().numpy()
                invsC = invs.C.cpu().numpy()
                invsF = invs.F.cpu().numpy()
                for batch_idx in range(self.batch_size):
                    fname = batch['file_name'][batch_idx]
                    assert fname == pool_set.im_idx[idx]
                    # Call Uncertainty Handler
                    uncertainty = self.uncertain_handler(preds)
                    cur_scene_pts = (scene_pts[:, -1] == batch_idx)
                    cur_inv = invsF[invsC[:, -1] == batch_idx]
                    output = preds[cur_scene_pts][cur_inv]  # (num_pts, Class)
                    uncertainty = self.uncertain_handler(output)
                    uncertainty = uncertainty.cpu().detach().numpy()
                    scores.append(uncertainty.item())

                    idx += 1
                    if idx >= len(pool_set.im_idx):
                        break
                if idx >= len(pool_set.im_idx):
                    break
        fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{trainer.local_rank}.json")
        with open(fname, "w") as f:
            json.dump(scores, f)

    def select_next_batch(self, trainer, active_set, selection_count):
        self.calculate_scores(trainer, active_set.pool_dataset)
        if trainer.distributed is False:
            fname = os.path.join(trainer.model_save_dir, "AL_record", "region_val_0.json")
            with open(fname, "r") as f:
                scores = json.load(f)
            # Comment: Reverse=True means the large the (former / better)
            selected_samples = list(zip(*sorted(zip(scores, active_set.pool_dataset.im_idx),
                                    key=lambda x: x[0], reverse=True)))[1][:selection_count]
            active_set.expand_training_set(selected_samples)
        else:
            dist.barrier()
            if trainer.local_rank == 0:
                scores = []
                for i in range(dist.get_world_size()):
                    fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{i}.json")
                    with open(fname, "r") as f:
                        scores.extend(json.load(f))
                # Comment: Reverse=True means the large the (former / better)
                selected_samples = list(zip(*sorted(zip(scores, active_set.pool_dataset.im_idx),
                                        key=lambda x: x[0], reverse=True)))[1][:selection_count]
                active_set.expand_training_set(selected_samples)


class RegionSoftmaxUncertaintySelector:

    def __init__(self, batch_size, num_workers, active_method):
        self.batch_size = batch_size
        self.num_workers = num_workers
        assert active_method in ['softmax_confidence', 'softmax_margin', 'softmax_entropy']
        if active_method == 'softmax_confidence':
            self.uncertain_handler = softmax_confidence
        if active_method == 'softmax_margin':
            self.uncertain_handler = softmax_margin
        if active_method == 'softmax_entropy':
            self.uncertain_handler = softmax_entropy

    def calculate_scores(self, trainer, pool_set):
        model = trainer.net
        model.eval()
        loader, idx = get_al_loader(trainer, pool_set, self.batch_size, self.num_workers)
        print(idx)

        scores = []
        tqdm_loader = tqdm(loader, total=len(loader))
        with torch.no_grad():
            for i_iter_test, batch in enumerate(tqdm_loader):
                # predict
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()

                inputs = batch['lidar']
                outputs = model(inputs)
                preds = outputs['final']
                invs = batch['inverse_map']
                all_labels = batch['targets_mapped']

                scene_pts = inputs.C.cpu().numpy()
                invsC = invs.C.cpu().numpy()
                invsF = invs.F.cpu().numpy()

                all_labels_F = all_labels.F.cpu().numpy()
                all_labels_C = all_labels.C.cpu().numpy()

                for batch_idx in range(self.batch_size):
                    fname = batch['file_name'][batch_idx]
                    assert fname == pool_set.im_idx[idx]

                    cur_scene_pts = (scene_pts[:, -1] == batch_idx)
                    cur_inv = invsF[invsC[:, -1] == batch_idx]
                    output = preds[cur_scene_pts][cur_inv]  # (num_pts, Class)
                    uncertainty = self.uncertain_handler(output)
                    uncertainty = uncertainty.cpu().detach().numpy()
                    cur_label = (all_labels_C[:, -1] == batch_idx)
                    cur_supvox = all_labels_F[cur_label]

                    # Groupby
                    key = pool_set.im_idx[idx]
                    df = pd.DataFrame({'id': cur_supvox, 'val': uncertainty})
                    df1 = df.groupby('id')['val'].agg(['count', 'mean']).reset_index()
                    table = df1[df1['id'].isin(pool_set.supvox[key])].drop(columns=['count'])
                    table['key'] = key
                    table = table.reindex(columns=['mean', 'key', 'id'])
                    region_score = list(table.itertuples(index=False, name=None))
                    scores.extend(region_score)

                    idx += 1
                    if idx >= len(pool_set.im_idx):
                        break
                if idx >= len(pool_set.im_idx):
                    break
        fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{trainer.local_rank}.json")
        with open(fname, "w") as f:
            json.dump(scores, f)

    def select_next_batch(self, trainer, active_set, selection_count):
        self.calculate_scores(trainer, active_set.pool_dataset)
        if trainer.distributed is False:
            fname = os.path.join(trainer.model_save_dir, "AL_record", "region_val_0.json")
            with open(fname, "r") as f:
                scores = json.load(f)
            selected_samples = sorted(scores, reverse=True)[:selection_count]
            active_set.expand_training_set(selected_samples)
        else:
            dist.barrier()
            if trainer.local_rank == 0:
                scores = []
                for i in range(dist.get_world_size()):
                    fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{i}.json")
                    with open(fname, "r") as f:
                        scores.extend(json.load(f))

                selected_samples = sorted(scores, reverse=True)[:selection_count]
                active_set.expand_training_set(selected_samples)
