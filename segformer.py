import torch
import torch.nn as nn
from semseg.models import *


def build_segformer(backbone, pretrained=True, num_classes=1): 
    """
    'MiT-B2'
    """
    model = eval('SegFormer')(
        backbone=backbone,
        num_classes=150
    )

    path_dict = {
        "MiT-B1": "/content/drive/MyDrive/2-專案/玉山-醫學影像切割比賽/models/segformer.b1.ade.pth",
        'MiT-B2': "/content/drive/MyDrive/2-專案/玉山-醫學影像切割比賽/models/segformer.b2.ade.pth",
        'MiT-B3': "/content/drive/MyDrive/2-專案/玉山-醫學影像切割比賽/models/segformer.b3.ade.pth"
    }

    if pretrained: 
        model.load_state_dict(torch.load(path_dict[backbone], map_location='cpu'))
    model.decode_head.linear_pred = nn.Conv2d(
        model.decode_head.linear_pred.in_channels,
        num_classes, 
        kernel_size=1
    )

    return model