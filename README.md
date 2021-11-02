# ReDAL: Region-based and Diversity-aware Active Learning for Point Cloud Semantic Segmentation
Official pytorch implementation of ["ReDAL: Region-based and Diversity-aware Active Learning for Point Cloud Semantic Segmentation. Wu et al. ICCV 2021."](https://arxiv.org/abs/2107.11769) [(presentation video)](https://www.youtube.com/watch?v=XJeb9kMxs5E)

In this work, we present **ReDAL**, a general active learning framework for point cloud semantic segmentation. By selecting only informative but diverse regions for label acquisition, the labeling effort can be hugely reduced.
Here shows a demo video.

https://user-images.githubusercontent.com/22555914/134839515-93f6523f-994f-4b22-b0ae-d8b554357f26.mp4


## Environmental Setup

- OS: Ubuntu 18.04
- CUDA: 11.1
- Install required packages (via conda)
  ```bash
  conda install pytorch==1.8.0 torchvision==0.9.0  cudatoolkit=11.1 -c pytorch -c conda-forge
  conda install pytorch-scatter -c pyg
  conda install scikit-learn=0.24.2
  conda install pyyaml=5.3.1
  conda install tqdm=4.61.1
  conda install pandas=1.3.2
  conda install pyntcloud -c conda-forge
  conda install plyfile -c conda-forge
  pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.2.0
  ```

Note: 
1. One might need to install `libsparsehash-dev` before installing torchsparse. For more information, please see the [official website](https://github.com/mit-han-lab/torchsparse).
2. We only support torchsparse@v1.2.0.
3. [torchsparse](https://anaconda.org/conda-forge/torchsparse) will automatically install the corresponding version of the package based on the the highest supported CUDA version of the nvidia-driver, where would be an alternative to install the torchsparse package.

## Data Preparation

This repo supports S3DIS and SemanticKITTI datasets. (Support for scannetv2 dataset is still in beta.)

Please see [this documentation](./data_preparation).

## Active Training Script

### A. Supervised Training (Upperbound)

- The script supports supervised training on the whole dataset, which is the upper-bound in our experiment.

```shell
CUDA_VISIBLE_DEVICES=X python3 train_supervision.py -n <dataset> -d <dataset-path> -p <stored-ckpt-path> \
                        -m <model-backbone> --train_batch_size <train-batch-size> --val_batch_size <val-batch-size> \
                        --ignore_idx <invalid-category-idx> --max-epoch <epochs> [--distributed_training]
```

### B. Scene-based Active Leraning (Baselines)
- The script supports multiple scene-based active learning strategies, which selects a batch of point cloud scans for label acquisition in an active iteration.
- Supported active\_method flags: `["random", "softmax_confidence", "softmax_margin", "softmax_entropy", "mc_dropout", "segment_entropy", "core_set"]`.

```shell
CUDA_VISIBLE_DEVICES=X python3 train_scene_active.py -n <dataset> -d <dataset-path> -p <stored-ckpt-directory> \
                        -m <model-backbone> --train_batch_size <train-batch-size> --val_batch_size <val-batch-size> \
                        --ignore_idx <invalid-category-idx> --training-epoch <epochs> --finetune-epoch <epochs> \
                        --active_method <valid-active-method> --max_iterations <AL iterations> \
                        --active_selection_size <number of scene labels per query> [--distributed_training]
```

### C. Region-based Active Learning

- The script supports multiple region-based active learning strategies, which selects a batch of divided sub-scene regions for label acquisition in an active iteration.
- Supported active\_method flags: `["random", "softmax_confidence", "softmax_margin", "softmax_entropy", "mc_dropout", "ReDAL"]`.
- **ReDAL** is our proposed method.

```shell
CUDA_VISIBLE_DEVICES=X python3 train_region_active.py -n <dataset> -d <dataset-path> -p <stored-ckpt-directory> \
                        -m <model-backbone> --train_batch_size <train-batch-size> --val_batch_size <val-batch-size> \
                        --ignore_idx <invalid-category-idx> --training-epoch <epochs> --finetune-epoch <epochs> \
                        --active_method <valid-active-method> --max_iterations <AL iterations> \
                        --active_percent <percent of labels per query> [--distributed_training]
```


## Citation

If you use this code, please cite the paper:

```
@inproceedings{wu2021redal,
  title={ReDAL: Region-based and Diversity-aware Active Learning for Point Cloud Semantic Segmentation},
  author={Wu, Tsung-Han and Liu, Yueh-Cheng and Huang, Yu-Kai and Lee, Hsin-Ying and Su, Hung-Ting and Huang, Ping-Chia and Hsu, Winston H},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15510--15519},
  year={2021}
}
```
