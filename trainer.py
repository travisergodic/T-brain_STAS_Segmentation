import torch
import torch.nn as nn
from tqdm import tqdm 
from sam.sam import SAM

class Trainer:
    def __init__(self, optim_dict, decay_fn, criterion, metric_dict, iter_hook, device): 
        self.decay_fn = decay_fn 
        self.device = device
        self.criterion = criterion
        self.metric_dict = metric_dict
        self.iter_hook = iter_hook
        self.optim_dict = optim_dict
    
    def _get_optimizer(self, model):
        optim_cls = self.optim_dict['optim_cls']
        optim_config = {k: self.optim_dict[k] for k in self.optim_dict if k != 'optim_cls'}
        if type(self.iter_hook).__name__[:3] == "SAM": 
            return SAM(model.parameters(), optim_cls, **optim_config)
        return optim_cls(model.parameters(), **optim_config)
        
    def _get_scheduler(self):
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda= self.decay_fn)
    
    def fit(self, model, train_loader, validation_loader, num_epoch, save_config, track=False):
        self.optimizer = self._get_optimizer(model)
        self.scheduler = self._get_scheduler()
        self.scaler = torch.cuda.amp.GradScaler()
        best_performance, best_epoch = -100, 0

        for epoch in range(1, num_epoch+1):
            print(f"Epoch [{epoch}/{num_epoch}] train_loss:{self._training_step(model, train_loader, self.criterion)}")
            val_loss = self._validation_step(model, validation_loader, self.criterion, self.metric_dict)
            # save model 
            if epoch % save_config["freq"] == 0:
                if isinstance(model, nn.DataParallel): 
                    torch.save(model.module, save_config["path"])
                else: 
                    torch.save(model, save_config["path"])
            # save best model
            if best_performance < -val_loss:
                best_performance, best_epoch = -val_loss, epoch
                if isinstance(model, nn.DataParallel): 
                    torch.save(model.module, save_config["best_path"])
                else: 
                    torch.save(model, save_config["best_path"])
            if track: 
                self.scheduler.step(val_loss.item())
            else: 
                self.scheduler.step()
        print(f"Best loss :{-best_performance} at {epoch} epoch.")

    def _training_step(self, model, train_loader, criterion): 
        model.train()
        loop = tqdm(train_loader)
        total_loss = 0 

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=self.device)
            targets = targets.type(torch.float).to(device=self.device)
            
            # run iter 
            iter_loss = self.iter_hook.run_iter(model, data, targets, self, criterion)
            total_loss += iter_loss * data.size(0)
            loop.set_postfix(loss=iter_loss)
        return total_loss/len(train_loader.dataset)

    @torch.no_grad()
    def _validation_step(self, model, validation_loader, criterion, metric_dict): 
        model.eval()
        test_loss = 0
        size = 0 
        metric_eval_dict = {k:0 for k in metric_dict}

        for data, targets in validation_loader:
            data = data.to(device=self.device)
            targets = targets.type(torch.float).to(device=self.device)
            predictions = model(data)
            loss = criterion(predictions, targets)
            test_loss += loss.item() * targets.size(0)
            size += targets.size(0)
            
            for metric_name in metric_dict: 
                metric_eval_dict[metric_name] += metric_dict[metric_name](predictions, targets)*targets.size(0)
                
        test_loss /= size

        print({k:(metric_eval_dict[k]/size).item() for k in metric_eval_dict})
        return test_loss