# basic
from sklearn.cluster import KMeans


# The core implementation of our diversity-aware selection algorithm.
def importance_reweight(scores, features, config):
    # sorted (first time)
    sorted_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    features = features[sorted_idx]
    selected_samples = sorted(scores, reverse=True)

    if config.trim_region is True:
        N = features.shape[0] * config.trim_rate
        features = features[:N]
        selected_samples = selected_samples[:N]
    # clustering
    m = KMeans(n_clusters=config.num_clusters, random_state=0)
    m.fit(features)
    clusters = m.labels_
    # importance re-weighting
    N = features.shape[0]
    importance_arr = [1 for _ in range(config.num_clusters)]
    for i in range(N):
        cluster_i = clusters[i]
        cluster_importance = importance_arr[cluster_i]
        scores[i][0] *= cluster_importance
        importance_arr[cluster_i] *= config.decay_rate
    # sorted (second time)
    selected_samples = sorted(scores, reverse=True)
    return selected_samples
