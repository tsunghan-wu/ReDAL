# Data Preprocessing \#2: Sub-Scene Region Division

## Description

In this step, we would like to divide a large-scale point cloud scan into some sub-scene regions as the fundamental label querying units in our ReDAL framework.

- Program Overview
    - C++ Program (VCCS algorithm)
    - Supported Dataset: S3DIS, SemanticKitti, Scannetv2

- TODO: Before this, see "Data Preprocessing \#1" first.

## Prerequisite

> Environment: Ubuntu 18.04 (only CPU is needed in this step)

1. Dependency Installation: [Point CLoud Library (PCL)](https://pointclouds.org/), [Boost C++ Library](https://www.boost.org/), [CMake](https://cmake.org/), [cnpy](https://github.com/rogersce/cnpy)

2. Build the project via CMake

We wrote a installtaion script to finish the above step.

```shell
cd src
bash install.sh
```

## Run

Example Script

- S3DIS Dataset

```shell
cd src/build
./supervoxel --dataset s3dis --input-path ~/Desktop/S3DIS_processed \
             --voxel-resolution 0.1 --seed-resolution 1 --color-weight 0.5 --spatial-weight 0.5
```

- Note: voxel-resolution, seed-resolution, color-weight and spatial-weight are hyperparameters in the original VCCS algorithm.
