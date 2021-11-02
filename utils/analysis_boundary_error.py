import os
import numpy as np


data_root = "/tmp3/tsunghan/S3DIS_processed"
exp_dir = "/home/master/09/tsunghan/spvcnn_softent_region_20210220"


def load_error(root_dir, AL_iter):
    fname = os.path.join(root_dir, f'selection_{AL_iter}', 'error.npy')
    if os.path.exists(fname):
        return np.load(fname).reshape(-1)
    else:
        return None


def load_boundary(name):
    area, _, fname = name.split("#")
    fname = os.path.join(data_root, area, "boundary", f'{fname}.npy')
    return np.load(fname).reshape(-1)


if __name__ == "__main__":
    result_dir = os.path.join(exp_dir, "result")
    for i in range(1, 11):
        boundary_num = 0
        non_boundary_num = 0
        boundary_error = 0
        non_boundary_error = 0

        for name in os.listdir(result_dir):
            error = load_error(os.path.join(result_dir, name), i)
            if error is None:
                continue
            boundary = load_boundary(name)
            assert error.shape == boundary.shape

            mask = (boundary == 1)

            boundary_num += mask.sum()
            non_boundary_num += (boundary == 0).sum()

            boundary_error += np.sum(error[mask])
            non_boundary_error += np.sum(error[~mask])
        # avg_boundary_error = round(boundary_error / boundary_num, 3)
        # avg_non_boundary_error = round(non_boundary_error / non_boundary_num, 3)
        print(f'[AL iter {i}]')
        print(f'- Avg boundary error: {boundary_error}')
        print(f'- Avg non-boundary error: {non_boundary_error}')
        # print(f'- Avg boundary error: {avg_boundary_error}')
        # print(f'- Avg non-boundary error: {avg_non_boundary_error}')
