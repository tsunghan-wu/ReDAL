import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.distributed as dist
from active_selection.utils import get_al_loader


class SegmentEntropySelector:

    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def calculate_scores(self, trainer, pool_set):
        pool_set.label_to_supvox()

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
                    output = preds[cur_scene_pts][cur_inv].argmax(1).cpu().numpy()

                    cur_label = (all_labels_C[:, -1] == batch_idx)
                    cur_supvox = all_labels_F[cur_label]

                    # Groupby
                    key = pool_set.im_idx[idx]
                    df = pd.DataFrame({'id': cur_supvox, 'val': output})

                    num_class = 13
                    possible_categories = [i for i in range(num_class)]
                    df['val'] = df['val'].astype(pd.CategoricalDtype(categories=possible_categories))
                    cat = pd.get_dummies(df['val'])
                    df1 = pd.concat([df, cat], sort=False, axis=1).drop(['val'], axis=1)
                    df2 = df1.groupby('id').agg('mean')
                    prob = df2.values
                    seg_ent = np.mean(-prob * np.log2(prob + 1e-12), axis=1)
                    scores.append(np.mean(seg_ent).item())

                    idx += 1
                    if idx >= len(pool_set.im_idx):
                        break
                if idx >= len(pool_set.im_idx):
                    break
        fname = os.path.join(trainer.model_save_dir, "AL_record", f"region_val_{trainer.local_rank}.json")
        with open(fname, "w") as f:
            json.dump(scores, f)

        pool_set.supvox_to_label()

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
