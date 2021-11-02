import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
from active_selection.utils import get_al_loader


class MCDropoutSelector:

    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_drop = 10

    def calculate_scores(self, trainer, pool_set):
        model = trainer.net
        model.eval()
        # Turn on Dropout
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
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
                # N-times prediction
                outputs = model(inputs)
                preds = torch.nn.functional.softmax(outputs['final'], dim=1)
                for i in range(1, self.n_drop):
                    outputs = model(inputs)
                    preds += torch.nn.functional.softmax(outputs['final'], dim=1)
                preds /= self.n_drop
                invs = batch['inverse_map']

                scene_pts = inputs.C.cpu().numpy()
                invsC = invs.C.cpu().numpy()
                invsF = invs.F.cpu().numpy()

                for batch_idx in range(self.batch_size):
                    fname = batch['file_name'][batch_idx]
                    assert fname == pool_set.im_idx[idx]

                    cur_scene_pts = (scene_pts[:, -1] == batch_idx)
                    cur_inv = invsF[invsC[:, -1] == batch_idx]
                    output = preds[cur_scene_pts][cur_inv]

                    # Softmax Entropy
                    score = torch.mean(-output * torch.log2(output + 1e-12), dim=1)
                    score = score.cpu().detach().numpy().mean()
                    scores.append(score.item())

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
                selected_samples = list(zip(*sorted(zip(scores, active_set.pool_dataset.im_idx),
                                        key=lambda x: x[0], reverse=True)))[1][:selection_count]
                active_set.expand_training_set(selected_samples)


class RegionMCDropoutSelector:

    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_drop = 10

    def calculate_scores(self, trainer, pool_set):
        model = trainer.net
        model.eval()
        # Turn on Dropout
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

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
                # N-times prediction
                outputs = model(inputs)
                preds = torch.nn.functional.softmax(outputs['final'], dim=1)
                for i in range(1, self.n_drop):
                    outputs = model(inputs)
                    preds += torch.nn.functional.softmax(outputs['final'], dim=1)
                preds /= self.n_drop
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
                    output = preds[cur_scene_pts][cur_inv]
                    uncertain = torch.mean(-output * torch.log2(output + 1e-12), dim=1)
                    uncertain = uncertain.cpu().detach().numpy()
                    cur_label = (all_labels_C[:, -1] == batch_idx)
                    cur_supvox = all_labels_F[cur_label]

                    # Groupby
                    key = pool_set.im_idx[idx]
                    df = pd.DataFrame({'id': cur_supvox, 'val': uncertain})
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

    def select_next_batch(self, trainer, active_set, selection_percent):
        self.calculate_scores(trainer, active_set.pool_dataset)
        if trainer.distributed is False:
            fname = os.path.join(trainer.model_save_dir, "AL_record", "region_val_0.json")
            with open(fname, "r") as f:
                scores = json.load(f)
            selected_samples = sorted(scores, reverse=True)
            active_set.expand_training_set(selected_samples, selection_percent)
        else:
            dist.barrier()
            if trainer.local_rank == 0:
                scores = []
                for i in range(dist.get_world_size()):
                    fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{i}.json")
                    with open(fname, "r") as f:
                        scores.extend(json.load(f))

                selected_samples = sorted(scores, reverse=True)
                active_set.expand_training_set(selected_samples, selection_percent)
