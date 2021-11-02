# Data prepatation

## Step 1: Dataset download

### A. S3DIS Dataset

1. Fill the [google form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform) to get the dataset download link (download **Stanford3dDataset_v1.2_Aligned_Version.zip**)

2. S3DIS Data Preprocessing
    1. extract "Stanford3dDataset_v1.2_Aligned_Version.zip".
    2. Modify dataset path in line 11, 12 in [s3dis\_data\_preparation.py](s3dis_data_preparation.py) and run it.
    3. Run the script below to prepare S3DIS dataset.
    ```shell
    python3 s3dis_data_preparation.py
    ```

After that, the file organization will be like:

```shell
S3DIS/
├── Area_1
│   ├── coords
│   │   ├── conferenceRoom_1.npy
│   │   ...
│   ├── labels
│   │   ├── conferenceRoom_1.npy
│   │   ...
│   └── rgb
│       ├── conferenceRoom_1.npy
│       ...
├── Area_2
...

```

### B. SemanticKITTI dataset

1. Download SemanticKITTI dataset from the [official website](http://www.semantic-kitti.org/dataset.html)
    - Download "KITTI Odometry Benchmark Velodyne point clouds (80 GB)" & "SemanticKITTI label data (179 MB)"

2. extract the archieve and organize the files as following:

```shell
SemanticKitti/
└── sequences
    ├── 00
    │   ├── labels
    │   │   ├── 000000.label
    │   │   ...
    │   └── velodyne
    │       ├── 000000.bin
    │       ...
    ├── 01
    ...
...
```

## Step 2: Sub-scene regions division

In this step, we would like to divide a large-scale point cloud scan into some sub-scene regions as the fundamental label querying units in our ReDAL framework.

- Program Overview
    - C++ Program (VCCS algorithm)
    - Supported Dataset: S3DIS, SemanticKitti, Scannetv2

### Prerequisite

> Environment: Ubuntu 18.04 (only CPU is needed in this step)

1. Dependency Installation: [Point CLoud Library (PCL)](https://pointclouds.org/), [Boost C++ Library](https://www.boost.org/), [CMake](https://cmake.org/), [cnpy](https://github.com/rogersce/cnpy)

2. Build the project via CMake

We wrote a installtaion script to finish the above step.

```shell
cd region_division/src
bash install.sh
```

### Run

Example Script

#### A. S3DIS Dataset

```shell
cd region_division/src/build
./supervoxel --dataset s3dis --input-path ~/Desktop/S3DIS_processed \
             --voxel-resolution 0.1 --seed-resolution 1 --color-weight 0.5 --spatial-weight 0.5
```

- Note: voxel-resolution, seed-resolution, color-weight and spatial-weight are hyperparameters in the original VCCS algorithm.

After that, your file orginization will be like:

```shell
S3DIS/
├── Area_1
│   ├── coords
│   │   ├── conferenceRoom_1.npy
│   │   ...
│   ├── labels
│   │   ├── conferenceRoom_1.npy
│   │   ...
│   ├── rgb
│   │   ├── conferenceRoom_1.npy
│   │   ...
│   └── supervoxel (new directory)
│       ├── conferenceRoom_1.npy
│       ...
├── Area_2
...
```

#### B. SemanticKITTI Dataset

```shell
cd region_division/src/build
./supervoxel --dataset s3dis --input-path ~/Desktop/SemanticKITTI/sequences \
             --voxel-resolution 0.5 --seed-resolution 10 --color-weight 0.0 --spatial-weight 1.0
```

- Note: voxel-resolution, seed-resolution, color-weight and spatial-weight are hyperparameters in the original VCCS algorithm.

After that, your file orginization will be like:

```shell
SemanticKitti/
└── sequences
    ├── 00
    │   ├── labels
    │   │   ├── 000000.label
    │   │   ...
    │   ├── supervoxel (new directory)
    │   │   ├── 000000.bin
    │   │   ...
    │   └── velodyne
    │       ├── 000000.bin
    │       ...
    ├── 01
    ...
...
```

## Step 3: Calculate point cloud properties.

In this step, we would like to calculate color difference (named as color discontinuity in our paper) and surface variation (named as structure complexity in our paper) for each point cloud scan. These point cloud properties provide additional information for us to measure the information score for a region and will be used in our ReDAL framework.

### A. S3DIS dataset

Please run both `gen_color_gradient.py` and `gen_surface_variation.py` with appropriate arguments.

After that, your file orginization will be like:

```shell
S3DIS/
├── Area_1
│   ├── boundary (new directory)
│   │   ├── conferenceRoom_1.npy
│   │   ...
│   ├── colorgrad (new directory)
│   │   ├── conferenceRoom_1.npy
│   │   ...
│   ├── coords
│   │   ├── conferenceRoom_1.npy
│   │   ...
│   ├── labels
│   │   ├── conferenceRoom_1.npy
│   │   ...
│   ├── rgb
│   │   ├── conferenceRoom_1.npy
│   │   ...
│   └── supervoxel
│       ├── conferenceRoom_1.npy
│       ...
├── Area_2
...
```


### B. SemanticKITTI dataset

Only `gen_surface_variation.py` are required to be run.

After that, your file orginization will be like:

```shell
SemanticKitti/
└── sequences
    ├── 00
    │   ├── boundary (new directory)
    │   │   ├── 000000.npy
    │   │   ...
    │   ├── labels
    │   │   ├── 000000.bin
    │   │   ...
    │   ├── supervoxel
    │   │   ├── 000000.bin
    │   │   ...
    │   └── velodyne
    │       ├── 000000.bin
    │       ...
    ├── 01
    ...
...
```

Now, you've finished all data preparation steps.
