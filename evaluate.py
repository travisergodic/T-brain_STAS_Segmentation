import os 
import torch 
import torch.nn as nn
from torchvision.transforms import Resize, InterpolationMode
from tqdm import tqdm
from PIL import Image
import numpy as np
import ttach as tta

class Evaluator:
    def __init__(self, models, image_transforms, device='cuda', activation=nn.Sigmoid()):
        models = models if isinstance(models, list) else [models]  
        assert len(models) == len(image_transforms)
        assert all([isinstance(model, nn.Module) for model in models])
        self.models = [
            nn.Sequential(
                model,
                nn.Identity() if activation is None else activation 
            ).to(device) for model in models
        ]
    
        self.image_transforms = image_transforms
        self.device = device

    def _wrap_TTA(self, tta_transform): 
        if tta_transform:
            return [
                tta.SegmentationTTAWrapper(model, tta_transform, merge_mode='mean').eval() 
                for model in self.models
            ]
        return [model.eval() for model in self.models]
    
    @torch.no_grad()
    def _predict(self, models, path, size, mask_mode='color'):
        mask = torch.zeros(size).to(self.device)
        img = Image.open(path).convert('RGB')
        reszie_fn = Resize(size, InterpolationMode.NEAREST)
        for model, image_transform in zip(models, self.image_transforms): 
            x = image_transform(img, None)[0].to(self.device) 
            prediction = model(x.unsqueeze(0))
            mask += reszie_fn(prediction).squeeze()
        mask = mask/len(models) > 0.5

        if mask_mode ==  'color': 
            mask = torch.where(mask, 255, 0)
            self._check_range(mask, [0, 255])
        elif mask_mode == 'class': 
            mask = torch.where(mask, 1, 0)
            self._check_range(mask, [0, 1])
        return mask
    
    def evaluate(self, image_paths, label_paths, metric, tta_transform=False):
        if type(image_paths) == str and type(label_paths) == str: 
            assert os.path.isdir(image_paths) and os.path.isdir(label_paths), f"{image_paths} or {label_paths} is not a directory!"
            image_paths = sorted([os.path.join(image_paths, basename) for basename in os.listdir(image_paths)])
            label_paths = sorted([os.path.join(label_paths, basename) for basename in os.listdir(label_paths)])
        assert len(image_paths) == len(label_paths)
        
        models = self._wrap_TTA(tta_transform)
        
        total_score = 0
        for image_path, label_path in tqdm(list(zip(image_paths, label_paths))): 
            gt = torch.from_numpy(np.load(label_path)['image']).to(self.device)
            mask = self._predict(models, image_path, gt.shape, 'class')
            total_score += getattr(Evaluator, metric)(mask, gt)
        return total_score/len(image_paths)            
    
    def make_prediction(self, paths, save_dir, tta_transform=False, mask_mode='color'): 
        assert mask_mode in ('color', 'class')
        print(f'mask mode:{mask_mode}')
        if type(paths) == str: 
            assert os.path.isdir(paths), "Input variable 'path' is not a directory!"
            img_dir = paths
            paths = [os.path.join(img_dir, basename) for basename in os.listdir(paths)]
            print(f"Find {len(paths)} files under {img_dir}")
        else: 
            assert all([os.path.isfile(path) for path in paths])
        
        origin_size = Image.open(paths[0]).size[::-1]
        print(f"Use {origin_size} as output!")

        models = self._wrap_TTA(tta_transform)
        
        for path in tqdm(paths): 
            mask = self._predict(models, path, origin_size, mask_mode)
            self._save(mask.cpu().numpy().astype(np.uint8), os.path.basename(path).split(".")[0] + ".png", save_dir)
        
    def _save(self, mask, basename, save_dir): 
        Image.fromarray(mask).save(os.path.join(save_dir, basename))
        
    def _check_range(self, mask, range_list): 
        res = torch.zeros_like(mask)
        for ele in range_list:
            res += (mask == ele).long()
        assert torch.all(res > 0)

    @staticmethod
    def dice_score(pred, gt, ep=1e-8):
        return ((2 * pred * gt).sum()/(pred.sum() + gt.sum() + ep)).item()