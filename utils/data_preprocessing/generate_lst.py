import os
import glob
import json
import numpy as np


data_root = '/work/patrickwu2/S3DIS_processed/'
split = ['1', '2', '3', '4', '6']
im_idx = []

for x in split:
    # READ SPLIT FILES
    fn = os.path.join(data_root, f"Area_{x}", "coords") + '/*.npy'
    im_idx.extend(glob.glob(fn))

train_file_list = [x.replace(data_root, '') for x in im_idx]

np.random.shuffle(train_file_list)
train_num = len(train_file_list) // 50
label_file = train_file_list[:train_num]
ulabel_file = train_file_list[train_num:]


# Save files
def save_list(lst, fname):
    with open(fname, "w") as f:
        json.dump(lst, f, indent=2)


save_list(label_file, "init_label_scan.json")
save_list(ulabel_file, "init_ulabel_scan.json")
