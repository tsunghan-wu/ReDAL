import os
import numpy as np
from plyfile import PlyData

exp_dir = "/home/master/09/tsunghan/spvcnn_softent_region_20210220"


def load_error(root_dir, AL_iter):
    fname = os.path.join(root_dir, f'selection_{AL_iter}', 'error.npy')
    if os.path.exists(fname):
        return np.load(fname).reshape(-1)
    else:
        return None


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


if __name__ == "__main__":
    result_dir = os.path.join(exp_dir, "result")
    for i in range(1, 11):
        all_tp = 0
        all_tn = 0
        all_error = 0
        # all_fp = 0
        selection_N = 0
        all_N = 0

        for name in os.listdir(result_dir):
            error = load_error(os.path.join(result_dir, name), i)
            if error is None:
                continue
            selection = load_selection(os.path.join(result_dir, name), i)
            assert error.shape == selection.shape

            tp = (selection == 1) & (error == 1)
            tn = (selection == 1) & (error == 0)
            # fp = (selection == 0) & (error == 1)

            all_tp += tp.sum()
            all_tn += tn.sum()
            # all_fp = fp.sum()

            selection_N += (selection == 1).sum()
            all_error += (error == 1).sum()
            all_N += error.shape[0]

        avg_tp = round(all_tp / selection_N, 3)
        avg_tn = round(all_tn / selection_N, 3)
        avg_err = round(all_error / all_N, 3)
        print(f'[AL iter {i}]')
        print(f'- Good selection: {avg_tp}')
        print(f'- Redundant selection: {avg_tn}')
        print(f'- Average Error: {avg_err}')
