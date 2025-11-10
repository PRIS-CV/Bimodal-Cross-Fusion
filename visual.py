from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import mmcv

config_file = './configs/dino/AO-DETR-DM_r50_pixray.py'
checkpoint_file = r'G:\Xray-detection-checkpoint\PIXray\AO-DETR-DM_v2\lr_mult_1_num_level_1\epoch_24.pth'

# config_file = './configs/dino/AO-DETR_r50_pixray.py'
# checkpoint_file = r'G:\Xray-detection-checkpoint\PIXray\AO-DETR\AO-DETR_v2\epoch_23.pth'

register_all_modules()
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img = mmcv.imread('091605136.png', channel_order='rgb')
result = inference_detector(model, img)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta
print(result)
visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=True,
    wait_time=0,
)
visualizer.show()