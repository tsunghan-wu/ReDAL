import os
import glob
import json
import numpy as np


data_root = "/tmp2/tsunghan/scannet_download/scans/"

train_lst_fname = "splits/scannetv2_train.txt"

with open(train_lst_fname, "r") as f:
    train_lst = f.read().split()

im_idx = []

for x in train_lst:
    fn = os.path.join(data_root, x, "coords.npy")
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
