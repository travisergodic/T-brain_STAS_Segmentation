## 安裝
```python
git clone https://github.com/travisergodic/STAS-segmentation.git
pip install -r requirements.txt
cd STAS-segmentation/
git clone https://github.com/davda54/sam.git
mkdir pretrained/
mkdir checkpoints/
```

## 前期準備
1. 下載預訓練模型: https://drive.google.com/drive/folders/1dg-VfFPqnkJuTeqRKZD5QJzhKqT32cx0?usp=sharing
2. 將預訓練模型檔放在 `pretrained/` 路徑下
3. 下載 `Annotations.zip`, `Train_Images.zip` 壓縮檔、將其解壓縮，並放在 `STAS-segmentation/` 路徑下: https://drive.google.com/drive/folders/1hG6MXfDO4QKYinOwdoS6xzHgGUSgF3f7

## 使用
1. **訓練**
```python
python train.py --config_file "train_config.py"
```
2. **評估**
```python
python test.py --model_paths "'./checkpoints/model_segformer.pt'"
```
3. **預測**
```python
python predict.py --model_paths "./models/model_path.pt" --target_dir "./data/Public_Image/" \
                --mask_mode "color" --do_tta "True"
```
