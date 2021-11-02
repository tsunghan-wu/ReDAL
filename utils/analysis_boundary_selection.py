import os
import numpy as np
from plyfile import PlyData


data_root = "/tmp3/tsunghan/S3DIS_processed"
exp_dir = "/home/master/09/tsunghan/spvcnn_random_region_20210220"


def load_selection(root_dir, AL_iter):
    fname = os.path.join(root_dir, f'selection_{AL_iter}', 'select.ply')
    plydata = PlyData.read(fname)
    r = plydata['vertex']['red'].reshape(-1, 1)
    g = plydata['vertex']['green'].reshape(-1, 1)
    b = plydata['vertex']['blue'].reshape(-1, 1)

    selection = np.zeros_like(r)
    mask = (r != 128) & (g != 128) & (b != 128)
    selection[mask] = 1
    return selection.reshape(-1)


def load_boundary(name):
    area, _, fname = name.split("#")
    fname = os.path.join(data_root, area, "boundary", f'{fname}.npy')
    return np.load(fname).reshape(-1)


if __name__ == "__main__":
    result_dir = os.path.join(exp_dir, "result")
    boundary_num = 0
    non_boundary_num = 0
    boundary_selection = 0
    non_boundary_selection = 0

    for i in range(1, 11):
        for name in os.listdir(result_dir):
            selection = load_selection(os.path.join(result_dir, name), i)
            if selection is None:
                continue
            boundary = load_boundary(name)
            assert selection.shape == boundary.shape

            mask = (boundary == 1)

            boundary_num += mask.sum()
            non_boundary_num += (boundary == 0).sum()

            boundary_selection += np.sum(selection[mask])
            non_boundary_selection += np.sum(selection[~mask])
        # avg_boundary_selection = round(boundary_selection / boundary_num, 3)
        # avg_non_boundary_selection = round(non_boundary_selection / non_boundary_num, 3)
        print(f'[AL iter {i}]')
        print(f'- Avg boundary selection: {boundary_selection}')
        print(f'- Avg non-boundary selection: {non_boundary_selection}')
