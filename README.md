# About Bimodal-Cross-Fusion
Code release for "HSV-RGB BI-MODAL FEATURE FUSION FOR PROHIBITED ITEM DETECTION IN X-RAY IMAGES"

Prohibited items detection in X-ray image has been widely applied in public places, items in luggage are often randomly stacked, resulting in blurred object edges in X-ray security images, which weakens the features of prohibited items and increases the difficulty of their detection. Most previous studies focus on processing RGB X-ray images. However, since the RGB color space is composed of red, green, and blue channels, it cannot adequately represent the grayscale information inherent in X-ray images. In contrast, the HSV color space describes color in terms of hue, saturation, and value, making it more suitable for modeling the pseudo-color distribution in X-ray image. Inspired by this, we introduce HSV color space and fuse features extracted from both RGB and HSV modalities for object detection. Specifically, we propose a Bi-modal Cross Fusion (BCF) module to integrate same level features from these two modalities. This module extracts valuable information from the HSV features and combines it with RGB features to enhance the detection performance of prohibited items. The experimental results on the PIXray dataset confirms the proposed approach’s effectiveness.

## Environment configuration
* For the specific installation tutorial of the environment, refer to [MMDetection official tutorial version 3.1.0](https://mmdetection.readthedocs.io/en/v3.1.0/get_started.html)
* Requirements
```python
python 3.9.17
pytorch 1.13.1
mmdet 3.1.0
mmcv 2.0.1
```

## Dataset
The experiments were conducted on two prohibited item datasets, PIXray and OPIXray. PIXray was evaluated using COCO metrics, and the corresponding COCO-format PIXray dataset can be obtained from the [AO-DETR](https://github.com/Limingyuan001/AO-DETR-test) method.


## Train
```
cd AO-DETR-DM
python train.py --config configs/dino/AO-DETR-DM_r50_pixray.py --work-dir checkpoint/ao-detr_dm/r50_pixray/train/
```
## Test
Taking the test based on the PIXray dataset as an example, assuming the corresponding checkpoint path is checkpoint/ao-detr_dm/r50_pixray/train/epoch_12.pth, enter the following command in the terminal:
```
python test.py --config configs/dino/AO-DETR-DM_r50_pixray.py --checkpoint checkpoint/ao-detr_dm/r50_pixray/train/epoch_12.pth
```
If you wish to perform visual analysis, you can enter the following command to save the test results:
```
python test.py --config configs/dino/AO-DETR-DM_r50_pixray.py --checkpoint checkpoint/ao-detr_dm/r50_pixray/train/epoch_12.pth  --out PklForConfusion/PIXray/AO-DETR-DM/epoch12.pkl
```
The trained checkpoint has been uploaded to [Baidu Netdisk](https://pan.baidu.com/s/1xnxUaXhsr5MkdhfXbNcnLQ?pwd=svej) with the extraction code:svej。

## Visual Analysis
To calculate the confusion matrix, enter the following command:
```
python tools/analysis_tools/confusion_matrix.py --config configs/dino/AO-DETR-DM_r50_pixray.py --prediction_path PklForConfusion/PIXray/AO-DETR-DM/epoch12.pkl --save_dir ./ConfusionResult/PIXray/AO-DETR-DM/
```

