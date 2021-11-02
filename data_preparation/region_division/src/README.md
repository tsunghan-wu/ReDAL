# Usage

## Compile

```bash
mkdir build
cd build
cmake ../
make
```
## Run

```bash
./supervoxel [args]
```

## Hyper parameter

1. SemanticKITTI

- Default Parameter

2. S3DIS

```
./supervoxel --dataset s3dis --input-path ~/Desktop/S3DIS_processed --voxel-resolution 0.1 --seed-resolution 1 --color-weight 0.5 --spatial-weight 0.5
```
