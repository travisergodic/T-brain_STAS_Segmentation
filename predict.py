import os
import argparse
import torch
from evaluate import Evaluator
from data import Test_Preprocessor
from configs.test_config import * 


def boolean_string(s):
    if s == 'False': 
        return False
    elif s == 'True': 
        return True   
    else:
        raise ValueError('Not a valid boolean string')

def make_prediction(model_paths, image_dir, mask_mode, do_tta):
    # load models
    models = [torch.load(model_path.strip()) for model_path in model_paths.split(",")]
    test_image_transforms =  [Test_Preprocessor(img_size) for img_size in test_img_size_list]

    # create predict_result directory 
    if os.path.isdir('./predict_result'): 
        import shutil 
        shutil.rmtree('./predict_result')
        print("Delete directory: predict_result/")
    os.mkdir('./predict_result')   
    print("Create directory: predict_result/")

    # make predictions
    evaluator = Evaluator(models, 
                          test_image_transforms, 
                          device=DEVICE, 
                          activation=activation)

    evaluator.make_prediction(image_dir, 
                              './predict_result', 
                              tta_fn if do_tta else False, 
                              mask_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic segmentation for medical dataset!")
    parser.add_argument("--model_paths", type=str)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--mask_mode", type=str)
    parser.add_argument("--do_tta", type=boolean_string)
    args = parser.parse_args()
    
    # evaluate
    make_prediction(args.model_paths, args.target_dir, args.mask_mode, args.do_tta)