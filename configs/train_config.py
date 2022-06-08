import torch
from torch import optim
import segmentation_models_pytorch as smp
import ttach as tta
import segformer

# device 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
distributed = False

# config
train_image_dir = '/content/STAS-segmentation/Train_Images/'
label_dir = '/content/STAS-segmentation/Annotations/'
seed = 14

# preprocess
ann_suffix = '.npz'
img_suffix = '.jpg'
train_img_size = (512, 512)
test_img_size = (512, 512)
h_flip_p=0.5
v_flip_p=0.5
affine_p=0.

# dataloader
train_ratio = 0.85
train_batch_size = 7
test_batch_size =  14
num_workers = 2

# train config 
num_epoch = 50
decay_fn = lambda n: 1
regularization_option = "normal"    # options: "sam", "mixup", "cutmix", "normal", "half_cutmix" 
optim_dict = {
    'optim_cls': optim.Adam, 
    'lr': 1e-4, 
    # 'weight_decay': 1e-3
}

## model 
checkpoint_path = None
model_dict = {
    'model_cls': segformer.build_segformer,
    'backbone': 'MiT-B2',
    'pretrained': True,
    'num_classes': 1
}

## save
save_config = {
    "path": './models/model_v2.pt',
    "best_path": './models/model_v2_best.pt',
    "freq": 5
}

## loss function 
class MixLoss:
    def __init__(self): 
        self.focal_loss = smp.utils.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2)
        self.dice_loss =  smp.utils.losses.DiceLoss(activation='sigmoid')
        
    def __call__(self, pred, targets): 
        return self.focal_loss(pred, targets) + self.dice_loss(pred, targets)

loss_fn = MixLoss()

## metric
metric_dict = {
    'IOU': smp.utils.metrics.IoU(threshold=0.5),
    'Accuracy': smp.utils.metrics.Accuracy(), 
    'DICE': smp.utils.losses.DiceLoss(activation='sigmoid')
}