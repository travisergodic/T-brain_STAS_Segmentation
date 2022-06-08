import os 
import glob
import time
import numpy as np 
import argparse
from batch_sampler import BatchSampler,RandomSampler
from data import StasDataset, Train_Preprocessor, Test_Preprocessor
from torch.utils.data import DataLoader
import torch.nn as nn
from hooks import *
from trainer import Trainer

def train(): 
    # do train test split & create val.txt file
    if not os.path.isfile("./val.txt"):         
        np.random.seed(seed)
        image_path_list = sorted(glob.glob(train_image_dir + "*" + img_suffix))
        np.random.shuffle(image_path_list)
        assert len(image_path_list) > 0
        split_index = int(len(image_path_list) * train_ratio)
        train_path_list = image_path_list[:split_index]
        test_path_list = image_path_list[split_index:]

        with open("./val.txt", "w") as f: 
            res = "\n".join([os.path.basename(test_path) for test_path in test_path_list])
            f.write(res)
            print("Create 'val.txt' file successfully!")
            
    # read val.txt file & create corresponding trainig and validation set 
    else: 
        print("'val.txt' file already exists!")
        with open("./val.txt", "r") as f: 
            image_path_list = [os.path.normpath(path) for path in glob.glob(train_image_dir + "*" + img_suffix)]
            assert len(image_path_list) > 0
            test_path_list = [os.path.normpath(os.path.join(train_image_dir, line.strip())) for line in f.readlines()]
            train_path_list = [image_path for image_path in image_path_list if image_path not in test_path_list]
        print("Read 'val.txt' file successfully!")
            
    
    print(f"Training set: {len(train_path_list)} images. \nValidation set: {len(test_path_list)} images. \n")
    
    # preprocesor 
    train_image_transform = Train_Preprocessor(None if do_multiscale else train_img_size,
                                         h_flip_p=h_flip_p,
                                         v_flip_p=v_flip_p)
    test_image_transform = Test_Preprocessor(test_img_size)
    
    # dataset
    train_dataset = StasDataset(train_path_list, label_dir, train_image_transform, ann_suffix)
    test_dataset = StasDataset(test_path_list, label_dir, test_image_transform, ann_suffix)
    
    # batchsampler & dataloader 
    if do_multiscale:
        batch_sampler = BatchSampler(RandomSampler(train_dataset),
                                     batch_size=train_batch_size,
                                     drop_last=False,
                                     multiscale_step=multiscale_step,
                                     img_sizes=multiscale_list)
        shuffle = False
    else: 
        batch_sampler = None
        shuffle = True
        
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        shuffle=shuffle
    )
   
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=test_batch_size, 
        num_workers=num_workers)

    # create model
    if checkpoint_path is not None: 
        model = torch.load(checkpoint_path).to(DEVICE)
        print(f'Load model from {checkpoint_path} successfully!')
    else:
        MODEL_CLS = model_dict.pop('model_cls')
        model = MODEL_CLS(**model_dict).to(DEVICE)
    
    # distributed 
    if distributed: 
        model = nn.DataParallel(model)

    # train_model
    start = time.time()
    
    ## get iter_hook_cls
    Iter_Hook_CLS = Iter_Hook_dict.get(regularization_option, Normal_Iter_Hook)
    
    print(f"Use iter hook of type <class {Iter_Hook_CLS.__name__}> during training!")
    train_pipeline = Trainer(optim_dict, decay_fn, loss_fn, metric_dict, Iter_Hook_CLS(), DEVICE)
    train_pipeline.fit(model, train_dataloader, test_dataloader, num_epoch, save_config)
    print(f"Training takes {time.time() - start} seconds!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic segmentation for medical dataset!")
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()

    assert os.path.isfile(os.path.join("./configs", args.config_file))
    with open(os.path.join("./configs", args.config_file), 'r') as f: 
        text = f.read()

    with open('./configs/config.py', 'w') as f:
        f.write(text)

    from configs.config import *
    train()  
    os.remove('./configs/config.py')