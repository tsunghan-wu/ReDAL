# basic
import os
import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from sklearn.metrics import pairwise_distances

# custom
from dataloader import get_dataset
from active_selection.utils import get_al_loader


class CoreSetSelector:

    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def calculate_scores(self, trainer, active_set):
        model = trainer.net
        model.eval()

        # ALL Dataset
        label_set = active_set.label_dataset
        pool_set = active_set.pool_dataset
        combine_lst = label_set.im_idx + pool_set.im_idx
        combine_set = get_dataset(name=trainer.args.name, data_root=None, imageset='custom-set', init_lst=combine_lst)
        loader, idx = get_al_loader(trainer, combine_set, self.batch_size, self.num_workers)
        print(idx)

        feature = []
        tqdm_loader = tqdm(loader, total=len(loader))
        with torch.no_grad():
            for i_iter_test, batch in enumerate(tqdm_loader):
                # predict
                for key, value in batch.items():
                    if 'name' not in key:
                        batch[key] = value.cuda()
                inputs = batch['lidar']
                outputs = model(inputs)
                feats = outputs['feat']
                featC = feats.C.cpu().numpy()
                featF = feats.F.cpu().numpy()

                for batch_idx in range(self.batch_size):
                    fname = batch['file_name'][batch_idx]
                    assert fname == combine_lst[idx]

                    feat = featF[featC[:, -1] == batch_idx].mean(axis=0).reshape(1, -1)
                    feature.append(feat)

                    idx += 1
                    if idx >= len(combine_set.im_idx):
                        break
                if idx >= len(pool_set.im_idx):
                    break
        feat_np = np.concatenate(feature, 0)
        fname = os.path.join(trainer.model_save_dir, "AL_record", f"coreset_feat_{trainer.local_rank}.npy")
        np.save(fname, feat_np)
        return combine_lst

    def _updated_distances(self, cluster_centers, features, min_distances):
        x = features[cluster_centers, :]
        dist = pairwise_distances(features, x, metric='euclidean')
        if min_distances is None:
            return np.min(dist, axis=1).reshape(-1, 1)
        else:
            return np.minimum(min_distances, dist)

    def _select_batch(self, features, selected_indices, N):
        new_batch = []
        min_distances = self._updated_distances(selected_indices, features, None)
        for _ in range(N):
            ind = np.argmax(min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in selected_indices
            min_distances = self._updated_distances([ind], features, min_distances)
            new_batch.append(ind)

        print('Maximum distance from cluster centers is %0.5f' % max(min_distances))
        return new_batch

    def select_next_batch(self, trainer, active_set, selection_count):
        combine_lst = self.calculate_scores(trainer, active_set)
        if trainer.distributed is False:
            fname = os.path.join(trainer.model_save_dir, "AL_record", "coreset_feat_0.npy")
            features = np.load(fname)
        else:
            dist.barrier()
            if trainer.local_rank == 0:
                feat_lst = []
                for i in range(dist.get_world_size()):
                    fname = os.path.join(trainer.model_save_dir, "AL_record", f"coreset_feat_{i}.npy")
                    feat_lst.append(np.load(fname))
                features = np.concatenate(feat_lst, 0)
        if trainer.local_rank == 0:
            label_num = len(active_set.label_dataset.im_idx)
            selected_indices = self._select_batch(features, list(range(label_num)), selection_count)
            active_set.expand_training_set([combine_lst[i] for i in selected_indices])
