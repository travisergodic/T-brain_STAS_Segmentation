## 安裝
```python
git clone https://github.com/travisergodic/STAS-segmentation.git
pip install segmentation-models-pytorch
pip install ttach
pip install kornia
pip install transformers
pip install einops

cd /content/STAS-segmentation
mkdir models
```

## 使用
1. 訓練
```python
python train.py --config_file "train_config.py"
```
2. 評估
```python
python test.py --model_paths "./models/model_path.pt"
```

3. 預測
```python
python predict.py --model_paths "./models/model_path.pt" --target_dir "./data/Public_Image/" \
                --mask_mode "color" --do_tta "True"
```
