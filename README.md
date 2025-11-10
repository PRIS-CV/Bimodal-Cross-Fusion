# AO-DETR-DM

## 方法介绍
* 本方法在AO-DETR方法的基础上进行改进，参考了CAFF-DINO方法的特征融合框架，设计了双模态特征融合模块（BFF），对图像RGB和HSV颜色空间的特征分层次进行融合。

## 环境配置
* 环境的具体安装教程参考[mmdetection官网教程3.1.0版本](https://mmdetection.readthedocs.io/en/v3.1.0/get_started.html)
* Requirements
```python
python 3.9.17
pytorch 1.13.1
mmdet 3.1.0
mmcv 2.0.1
```

## 训练与评估测试
实验在PIXray和OPIXray两个违禁物品数据集上展开，PIXray使用COCO指标进行评估，对应的COCO格式的PIXray是数据集可以在[AO-DETR](https://github.com/Limingyuan001/AO-DETR-test)方法中获取
### 训练
此处以使用PIXray数据集进行训练为例，在终端输入以下指令：
```
cd AO-DETR-DM
python train.py --config configs/dino/AO-DETR-DM_r50_pixray.py --work-dir checkpoint/ao-detr_dm/r50_pixray/train/
```
### 测试
* 此处以基于PIXray数据集进行测试为例，这里假设对应的checkpoint路径为checkpoint/ao-detr_dm/r50_pixray/train/epoch_12.pth,在终端输入以下指令：
```
python test.py --config configs/dino/AO-DETR-DM_r50_pixray.py --checkpoint checkpoint/ao-detr_dm/r50_pixray/train/epoch_12.pth
```
* 若要进行可视化分析，则可输入以下指令，保存测试结果：
```
python test.py --config configs/dino/AO-DETR-DM_r50_pixray.py --checkpoint checkpoint/ao-detr_dm/r50_pixray/train/epoch_12.pth  --out PklForConfusion/PIXray/AO-DETR-DM/epoch12.pkl
```
* 训练所得的checkpoint已上传至[百度云盘](https://pan.baidu.com/s/1xnxUaXhsr5MkdhfXbNcnLQ?pwd=svej),提取码：svej。

### 可视化分析
* 计算混淆矩阵，可输入以下指令
```
python tools/analysis_tools/confusion_matrix.py --config configs/dino/AO-DETR-DM_r50_pixray.py --prediction_path PklForConfusion/PIXray/AO-DETR-DM/epoch12.pkl --save_dir ./ConfusionResult/PIXray/AO-DETR-DM/
```

