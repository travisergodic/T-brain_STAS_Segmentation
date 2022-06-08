from tkinter import N
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from .head import SegFormerHead


class SegFormer(nn.Module):
    @classmethod
    def load_pretrained(backbone, num_classes): 
        model = SegFormer(backbone, num_classes=19)
        path_dict = {
            "MiT-B1": "/content/drive/MyDrive/2-專案/玉山-醫學影像切割比賽/models/segformer.b1.ade.pth",
            'MiT-B2': "/content/drive/MyDrive/2-專案/玉山-醫學影像切割比賽/models/segformer.b2.ade.pth",
            'MiT-B3': "/content/drive/MyDrive/2-專案/玉山-醫學影像切割比賽/models/segformer.b3.ade.pth"
        }


        model.load_state_dict(torch.load(path_dict.get(backbone), map_location='cpu'))
        model.decode_head.linear_pred = nn.Conv2d(
            model.decode_head.linear_pred.in_channels,
            num_classes, kernel_size=1
        )
        return model

    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        backbone, variant = backbone.split('-')
        self.backbone = eval(backbone)(variant)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


if __name__ == '__main__':
    model = SegFormer('MiT-B0')
    x = torch.zeros(1, 3, 512, 512)
    y = model(x)
    print(y.shape)