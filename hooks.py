from abc import ABC, abstractmethod
import torch
from utils import mixup_data, cutmix_data, half_cutmix_data, mixup_criterion
from kornia.augmentation import RandomErasing

__all__ = ['Iter_Hook_dict', 'Normal_Iter_Hook']


class Base_Iter_Hook(ABC): 
    @abstractmethod
    def run_iter(self, model, data, targets, trainer, criterion): 
        pass


class SAM_Iter_Hook(Base_Iter_Hook): 
    def run_iter(self, model, data, targets, trainer, criterion):
        # first forward-backward pass
        loss = criterion(model(data), targets)  # use this loss for any training statistics
        loss.backward()
        trainer.optimizer.first_step(zero_grad=True)
        
        # second forward-backward pass
        criterion(model(data), targets).backward()  # make sure to do a full forward pass
        trainer.optimizer.second_step(zero_grad=True)
        return loss.item()
        
    
class Mixup_Iter_Hook(Base_Iter_Hook):
    def run_iter(self, model, data, targets, trainer, criterion):
        with torch.cuda.amp.autocast():
            # forward
            mixed_data, targets_a, targets_b, lam = mixup_data(data, targets)
            predictions = model(mixed_data)
            loss = mixup_criterion(targets_a, targets_b, lam)(criterion, predictions)
     
            # backward
            trainer.optimizer.zero_grad()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        return loss.item() 
     

class Cutmix_Iter_Hook(Base_Iter_Hook): 
    def run_iter(self, model, data, targets, trainer, criterion):
        with torch.cuda.amp.autocast():
            # forward
            data, targets = cutmix_data(data, targets)
            predictions = model(data)
            loss = criterion(predictions, targets)
            
            # backward
            trainer.optimizer.zero_grad()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        return loss.item()           


class Half_Cutmix_Iter_Hook(Base_Iter_Hook): 
    def run_iter(self, model, data, targets, trainer, criterion):
        with torch.cuda.amp.autocast():
            # forward
            data, targets = half_cutmix_data(data, targets)
            predictions = model(data)
            loss = criterion(predictions, targets)
            
            # backward
            trainer.optimizer.zero_grad()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        return loss.item()


class Normal_Iter_Hook(Base_Iter_Hook): 
    def run_iter(self, model, data, targets, trainer, criterion):
        with torch.cuda.amp.autocast():
            # forward
            predictions = model(data)
            loss = criterion(predictions, targets)
            # backward
            trainer.optimizer.zero_grad()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        return loss.item()
    
    
class Cutout_Iter_Hook(Base_Iter_Hook): 
    def __init__(self): 
        self.aug = RandomErasing(scale=(0.02, 0.1), 
                                 ratio=(0.3, 3.3), 
                                 value=0.0, 
                                 same_on_batch=False, 
                                 p=0.5, keepdim=False)
    
    def run_iter(self, model, data, targets, trainer, criterion):
        with torch.cuda.amp.autocast():
            # forward
            data, targets = self.aug(data), self.aug(targets, params=self.aug._params)
            predictions = model(data)
            loss = criterion(predictions, targets)
            
            # backward
            trainer.optimizer.zero_grad()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
        return loss.item()        


Iter_Hook_dict = {
    "sam": SAM_Iter_Hook, 
    "mixup": Mixup_Iter_Hook, 
    "cutmix": Cutmix_Iter_Hook, 
    "half_cutmix": Half_Cutmix_Iter_Hook,
    "cutout": Cutout_Iter_Hook,
    "normal": Normal_Iter_Hook
}