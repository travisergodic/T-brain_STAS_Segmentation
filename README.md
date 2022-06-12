## 前期準備
```python
git clone https://github.com/travisergodic/STAS-segmentation.git
cd STAS-segmentation/
pip install -r requirements.txt
git clone https://github.com/davda54/sam.git
mkdir pretrained/
mkdir checkpoints/
mkdir final_models/
```

## 檔案下載
1. **下載預訓練模型**：
   + 下載 **SegFormer** 預訓練模型檔：https://drive.google.com/drive/folders/1dg-VfFPqnkJuTeqRKZD5QJzhKqT32cx0?usp=sharing
   + 將檔案放在 `pretrained/` 路徑
2. **下載資料**：
   + 下載`Annotations.zip`, `Train_Images.zip `, `Private_Image.zip`  ：https://drive.google.com/drive/folders/1hG6MXfDO4QKYinOwdoS6xzHgGUSgF3f7
   + 解壓縮檔案
   + 放在 `STAS-segmentation/` 路徑
3. **下載比賽所使用模型**：
   + 下載路徑：https://drive.google.com/drive/u/0/folders/1yCYWUaCtR6ODbRGXnPmRyaf1FYSznA3Z
   + 將 `segformer_b2.pt`,  `unetpp_efficientnetv2_large.pt` 放在 `final_models` 路徑

## 使用方法
1. **訓練**

   ```python
   python train.py --config_file "train_config_segformer.py"
   ```
2. **評估**

   ```python
   python test.py --model_paths "./checkpoints/model_segformer.pt"
   ```
3. **預測**

   ```python
   python predict.py --model_paths "./checkpoints/model_segformer.pt" --target_dir "./Public_Image/" \
                   --mask_mode "color" --do_tta "True"
   ```



## 使用比賽的模型做預測

1. **UNET++ & efficientnetv2 large**:  0.895215 (private dataset Dice Score)

   + 請先修改 `configs/test_config.py` 腳本中的第五行為 `test_img_size_list = [(384, 384)]`

   + 執行指令

     ```python
     python predict.py --model_paths "./final_model/unetpp_efficientnetv2_large.pt" --target_dir "./Public_Image/" \
                     --mask_mode "color" --do_tta "True"
     ```

     

2. **SegFormer B2**: 0.909592 (private dataset Dice Score)

   + 請先修改 `configs/test_config.py` 腳本中的第五行為 `test_img_size_list = [(512, 512)]`

   + 執行指令

     ```python
     python predict.py --model_paths "./final_models/segformer_b2.pt" --target_dir "./Public_Image/" \
                     --mask_mode "color" --do_tta "True"
     ```

3. **Ensemble**: 0.906326 (private dataset Dice Score)

   + 請先修改 `configs/test_config.py` 腳本中的第五行為 `test_img_size_list = [(384, 384), (512, 512)]`

   + 執行指令

     ```python
     python predict.py --model_paths "./final_model/unetpp_efficientnetv2_large.pt, ./final_models/segformer_b2.pt" --target_dir "./Public_Image/" \
                     --mask_mode "color" --do_tta "True"
     ```

     

    



