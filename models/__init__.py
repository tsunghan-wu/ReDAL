def get_model(name, model, num_classes):
    assert name in ['s3dis', 'semantic_kitti', 'scannet']
    assert model in ['spvcnn', 'minkunet']

    if name == 's3dis':
        from models.s3dis import SPVCNN, MinkUNet
        if model == 'spvcnn':
            net = SPVCNN(num_classes=num_classes, cr=1.0, pres=0.05, vres=0.05)
        elif model == 'minkunet':
            net = MinkUNet(num_classes=num_classes, cr=1.0)

    elif name == 'semantic_kitti':
        from models.semantic_kitti import SPVCNN, MinkUNet
        if model == 'spvcnn':
            net = SPVCNN(num_classes=num_classes, cr=1.0, pres=0.05, vres=0.05)
        elif model == 'minkunet':
            net = MinkUNet(num_classes=num_classes, cr=1.0)
    elif name == 'scannet':
        from models.scannet import SPVCNN, MinkUNet
        if model == 'spvcnn':
            net = SPVCNN(num_classes=num_classes, cr=1.0, pres=0.05, vres=0.05)
        elif model == 'minkunet':
            net = MinkUNet(num_classes=num_classes, cr=1.0)
    return net
