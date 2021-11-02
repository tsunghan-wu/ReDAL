import json
import numpy as np
import os
from os.path import join
from tqdm import tqdm


dataset_path = "/work/patrickwu2/SemanticKitti/sequences"


seq_list = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
train_file_list = []

label_file = []
ulabel_file = []

for seq_id in tqdm(seq_list):
    seq_path = join(dataset_path, seq_id)
    pc_path = join(seq_path, 'velodyne')
    for pc_file in os.listdir(pc_path):
        relative_fname = os.path.join(seq_id, 'velodyne', pc_file)
        train_file_list.append(relative_fname)
        # train_file_list.append((seq_id, pc_file[:-4]))


np.random.seed(1126)
np.random.shuffle(train_file_list)
train_num = len(train_file_list) // 200
label_file = train_file_list[:train_num]
ulabel_file = train_file_list[train_num:]

# Save files
def save_list(lst, fname):
    with open(fname, "w") as f:
        json.dump(lst, f, indent=2)


save_list(label_file, "../../dataloader/semantic_kitti/init_data/init_label_scan05.json")
save_list(ulabel_file, "../../dataloader/semantic_kitti/init_data/init_ulabel_scan05.json")
