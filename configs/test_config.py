import ttach as tta
import torch
import torch.nn as nn 

test_img_size_list = [(512, 512)] 
tta_fn = tta.aliases.flip_transform()
activation = nn.Sigmoid()
image_dir = './Train_Images/'
label_dir = './Annotations/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'