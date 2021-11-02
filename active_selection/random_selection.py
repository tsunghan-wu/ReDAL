import random


class RandomSelector:

    def select_next_batch(self, trainer, active_set, selection_count):
        scores = []
        for i in range(len(active_set.pool_dataset.im_idx)):
            scores.append(random.random())
        selected_samples = list(zip(*sorted(zip(scores, active_set.pool_dataset.im_idx),
                                            key=lambda x: x[0], reverse=True)))[1][:selection_count]
        active_set.expand_training_set(selected_samples)


class RegionRandomSelector:

    def select_next_batch(self, trainer, active_set, selection_percent):
        if trainer.local_rank == 0:
            scores = []
            # Give each supervoxel a random score
            for key in active_set.pool_dataset.supvox:
                for supvox_id in active_set.pool_dataset.supvox[key]:
                    score = random.random()
                    item = (score, key, supvox_id)
                    scores.append(item)
            # Sort the score
            selected_samples = sorted(scores, reverse=True)
            active_set.expand_training_set(selected_samples, selection_percent)
