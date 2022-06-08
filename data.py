import os
import random
import numpy as np
from PIL import Image
import torch 
import torch.nn as nn
from torch.utils.data import Dataset 
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F


class StasDataset(Dataset):
    def __init__(self, image_path_list, label_dir, image_transform, ann_suffix):
        super().__init__()
        self.image_path_list = image_path_list
        self.label_dir = label_dir
        self.image_transform = image_transform 
        self.ann_suffix = ann_suffix
        self.resize_image_fn_dict, self.resize_mask_fn_dict = dict(), dict()
    
    def __getitem__(self, item):
        if type(item) == int: 
            index, size = item, None
        elif type(item) == list or type(item) == tuple:
            index, size = item
        
        image = Image.open(self.image_path_list[index]).convert('RGB')
        
        if self.ann_suffix == '.png':
            label_path = os.path.join(
                self.label_dir, 
                os.path.basename(self.image_path_list[index]).split(".")[0] + self.ann_suffix
            )
            
            label = torch.from_numpy(np.array(Image.open(label_path))).unsqueeze(dim=0)
            
        elif self.ann_suffix == '.npz': 
            label_path = os.path.join(
                self.label_dir, 
                'label_' + os.path.basename(self.image_path_list[index]).split(".")[0] + self.ann_suffix
            )
            
            label = torch.from_numpy(np.load(label_path)['image']).unsqueeze(dim=0)

        if size is not None:
            if size not in self.resize_fn_dict: 
                self.resize_image_fn_dict[size] = transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR)
                self.resize_mask_fn_dict[size] = transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST)
            image = self.resize_image_fn_dict[size](image)
            label = self.resize_mask_fn_dict[size](label)
        
        # if multiscale_list is None, image transform should not contain resize function 
        if self.image_transform is not None:
            image, label = self.image_transform(image, label)
        return image, label

    def __len__(self):
        return len(self.image_path_list)



class Train_Preprocessor(nn.Module): 
    def __init__(self, img_size=None, h_flip_p=0.5, v_flip_p=0.5):
        super().__init__()
        if img_size is not None: 
            self.img_size = img_size
            self.resize_image = transforms.Resize(self.img_size, interpolation=InterpolationMode.BILINEAR) 
            self.resize_mask = transforms.Resize(self.img_size, interpolation=InterpolationMode.NEAREST) 
        else: 
            self.resize_image = nn.Identity()
            self.resize_mask = nn.Identity()
            
        self.jitter = transforms.ColorJitter(0.15, 0.15)
        self.blur = transforms.GaussianBlur((1, 3))
        
        self.h_flip_p = h_flip_p
        self.v_flip_p = v_flip_p
        
        self.preprocess = transforms.Compose(
            [
              transforms.ToTensor(),
              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )
    
    @torch.no_grad()
    def forward(self, img, label): 
        # random crop
        W, H = img.size
        w, h = random.randint(int(0.90*W), W), random.randint(int(0.90*H), H)
        i, j = random.randint(0, H-h), random.randint(0, W-w)
        img = F.crop(img, i, j, h, w)
        label = F.crop(label, i, j, h, w)

        # resize & color transform 
        img = self.blur(self.jitter(self.resize_image(img)))
        label = self.resize_mask(label)

        # Random horizontal flipping
        if random.random() < self.h_flip_p:
            img = F.hflip(img)
            label = F.hflip(label)

        # Random vertical flipping
        if random.random() < self.v_flip_p:
            img = F.vflip(img)
            label = F.vflip(label)       
            
        return self.preprocess(img), label


class Test_Preprocessor(nn.Module): 
    def __init__(self, img_size=None):
        super().__init__()
        if img_size is not None:
            self.resize_image = transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR)
            self.resize_mask = transforms.Resize(img_size, interpolation=InterpolationMode.NEAREST)
        else: 
            self.resize_image = nn.Identity()
            self.resize_mask = nn.Identity()
            
        self.preprocess = transforms.Compose(
            [ 
              transforms.ToTensor(),
              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )

    @torch.no_grad()
    def forward(self, img, label): 
        return self.preprocess(self.resize_image(img)), None if label is None else self.resize_mask(label)