import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from plyfile import PlyData, PlyElement


data_root = "/tmp2/tsunghan/S3DIS_processed"
exp_dir = "/home/master/09/tsunghan/spvcnn_softent_region_20210221"
sim_root = "similarity_pair/"

def load_selection(root_dir, AL_iter, data_root):
    fname = os.path.join(root_dir, f'selection_{AL_iter:02d}.pkl')
    with open(fname, 'rb') as f:
        raw_data = pickle.load(f)
    selection = []
    for item in raw_data:
        _, path, supvox_id = item
        basename = '/'.join(path.split('/')[-3:])
        new_path = os.path.join(data_root, basename)
        selection.append([new_path, supvox_id])
    return selection


def load_feature(root_dir, AL_iter, world_size=1):
    feat_lst = []
    for i in range(world_size):
        fname = os.path.join(root_dir, 'Region_Feature', f'region_feature_AL{AL_iter}_rank{i}.npy')
        feat_lst.append(np.load(fname))
    features = np.concatenate(feat_lst, 0)
    return features


def get_label(selection):
    label = []
    for fn, supvox_id in selection:
        labels_path = fn.replace('coords', 'labels')
        supvox_path = fn.replace('coords', 'supervoxel')
        labels = np.load(labels_path).reshape(-1)
        supvox = np.load(supvox_path).reshape(-1)
        valid = (supvox == supvox_id)
        labels = labels[valid]
        # get max
        u, c = np.unique(labels, return_counts=True)
        y = u[c == c.max()]
        label.append(y)


def to_ply(pos, colors, ply_fname):
    # to ply
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    el = PlyElement.describe(ply_array, 'vertex')
    PlyData([el], text=False).write(ply_fname)


def save_ply(al_sim_root, selection, sim_pair):
    idx1, idx2 = sim_pair
    fn = selection[idx1][0]

    supvox_path = fn.replace('coords', 'supervoxel')
    rgb_path = fn.replace('coords', 'rgb')
    coords = np.load(fn)
    rgb = np.load(rgb_path)
    supvox = np.load(supvox_path)
    # save rgb
    basename = '#'.join(fn.split('/')[-3:]).replace('.npy', '')
    basename_dir = os.path.join(al_sim_root, basename)
    os.makedirs(basename_dir, exist_ok=True)
    to_ply(coords, rgb, os.path.join(basename_dir, 'rgb.ply'))
    # save supervoxel pair
    supvox_rgb = np.ones_like(rgb) * 128
    supvox_rgb[(supvox == selection[idx1][1])] = np.array([255, 0, 0])
    supvox_rgb[(supvox == selection[idx2][1])] = np.array([255, 0, 0])
    to_ply(coords, supvox_rgb, os.path.join(basename_dir, 'supvox.ply'))


if __name__ == "__main__":
    os.makedirs(sim_root, exist_ok=True)
    for i in range(1, 11):
        al_sim_root = os.path.join(sim_root, f'AL_{i}')
        os.makedirs(al_sim_root, exist_ok=True)
        features = load_feature(exp_dir, i)
        selection = load_selection(exp_dir, i, data_root)
        dist = euclidean_distances(features, features)
        N = dist.shape[0]
        min_dist = 0.5
        sim_result = []

        for i in range(N):
            index_array = np.argpartition(dist[i], kth=2)
            most_sim_idx = index_array[1]
            if dist[i, most_sim_idx] < min_dist:
                print(selection[i], selection[most_sim_idx], dist[i, most_sim_idx])
                if selection[i][0] == selection[most_sim_idx][0]:
                    sim_result.append([i, most_sim_idx])
        # supervoxel visualization
        for sim_pair in sim_result:
            save_ply(al_sim_root, selection, sim_pair)

        # label = get_label(selection)
        # feature shape: (N, 2)
        # X_embedded = TSNE(init='pca', n_components=2).fit_transform(features)
        # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)
        # plt.savefig(f'AL_feature_{i:02d}.png')
        # plt.close()
