def get_dataset(name, data_root, imageset, voxel_size=0.05, init_lst=None):
    from dataloader.s3dis.dataset import Stanford3DDataset
    from dataloader.scannet.dataset import ScannetDataset
    from dataloader.semantic_kitti.dataset import SemKITTI
    if name == 's3dis':
        dataset = Stanford3DDataset(data_root, voxel_size, imageset=imageset, init_lst=init_lst)
    elif name == 'semantic_kitti':
        dataset = SemKITTI(data_root, voxel_size, imageset=imageset, init_lst=init_lst)
    elif name == 'scannet':
        dataset = ScannetDataset(data_root, voxel_size, imageset=imageset, init_lst=init_lst)
    else:
        raise NotImplementedError("Only Support s3dis and semantic_kitti dataset!")
    return dataset


def get_active_dataset(args, mode='scan'):
    if mode == 'scan':
        from dataloader.s3dis.active_dataset import ActiveStanford3DDataset
        from dataloader.scannet.active_dataset import ActiveScannetDataset
        from dataloader.semantic_kitti.active_dataset import ActiveSemKITTI
        if args.name == 's3dis':
            dataset = ActiveStanford3DDataset(args)
        elif args.name == 'semantic_kitti':
            dataset = ActiveSemKITTI(args)
        elif args.name == 'scannet':
            dataset = ActiveScannetDataset(args)
        else:
            raise NotImplementedError("Only Support s3dis and semantic_kitti dataset!")
    elif mode == 'region':
        from dataloader.semantic_kitti.region_active_dataset import RegionActiveSemKITTI
        from dataloader.s3dis.region_active_dataset import RegionActiveStanford3DDataset
        from dataloader.scannet.region_active_dataset import RegionActiveScannet
        if args.name == 's3dis':
            dataset = RegionActiveStanford3DDataset(args)
        elif args.name == 'semantic_kitti':
            dataset = RegionActiveSemKITTI(args)
        elif args.name == 'scannet':
            dataset = RegionActiveScannet(args)
        else:
            raise NotImplementedError("Only Support s3dis and semantic_kitti dataset!")
    return dataset
