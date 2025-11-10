# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple
import cv2
import torch
from torch import Tensor, nn
from torch.nn.init import normal_
import numpy as np
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType,ConfigType
from ..layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from ..layers import BCF
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from .dinov2_2 import DINOv2


# dinov2 通过在def forward_transformer（）中self.label_embeddings=self.dn_query_generator.label_embedding取消了xavier初始化


@MODELS.register_module()
class DINO_DM(DINOv2):
    r"""Implementation of `DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator. Defaults to `None`.
    """

    def __init__(self,
                 *args,
                 dn_cfg: OptConfigType = None,
                 backbone_hsv: ConfigType,
                 **kwargs) -> None:
        super().__init__(*args,dn_cfg=dn_cfg, **kwargs)
        # assert self.as_two_stage, 'as_two_stage must be True for DINO'
        # assert self.with_box_refine, 'with_box_refine must be True for DINO'

        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)
        self.backbone_hsv = MODELS.build(backbone_hsv)


    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        # self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
        # 初始化BCF,根据neck输入通道数进行初始化每一层的BCF
        bcf_list = []
        for c in [512, 1024, 2048]:
            bcf_cfg = dict(
                type='BCF',
                bs=4,
                embed_dim=c,
                num_heads=8,
                hidden_dim=c*2,
                dropout=0.1
            )
            bcf_list.append(MODELS.build(bcf_cfg))
        # for c in [512, 1024, 2048]:
        #     bcf_cfg = dict(
        #         type='BCF',
        #         in_channels=c,
        #     )
        #     bcf_list.append(MODELS.build(bcf_cfg))
        self.bcf = nn.ModuleList(bcf_list)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        # nn.init.xavier_uniform_(self.label_embeddings.weight)
        normal_(self.level_embed)
        for bcf in self.bcf:
            for p in bcf.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        # # 冻结hsv特征提取
        # for param in self.backbone_hsv.parameters():
        #     param.requires_grad = False 

    def extract_feat(self, batch_inputs:Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        # 提取RGB图像特征
        x = self.backbone(batch_inputs)
        # 转换为HSV空间，提取HSV图像特征
        hsv_input = rgb_to_hsv(batch_inputs).to('cuda')
        x_hsv = self.backbone_hsv(hsv_input) 
        
        
        # 获取每一层特征，分别进行特征融合操作
        x_fuse = []
        for i in range(0,len(x)):
            fused_feat = self.bcf[i](x[i],x_hsv[i])
            # print(f"\nFused Feature Level {i}:")
            # print(f"Shape: {fused_feat.shape}")
            # print(f"Range: [{fused_feat.min().item():.3f}, {fused_feat.max().item():.3f}]")
            x_fuse.append(fused_feat)
            
        # 将list转化为tuple(Tensor)
        x_fuse = tuple(x_fuse)
        # 若有neck，则进行neck操作
        if self.with_neck:
            x_fuse = self.neck(x_fuse)
            
        return x_fuse




def rgb_to_hsv(batch_inputs):
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    
    batch_inputs_np = batch_inputs.permute(0, 2, 3, 1).cpu().numpy()  # (bs, H, W, 3)
    batch_inputs_np = batch_inputs_np * std + mean  # 反归一化
    
    batch_inputs_np = np.clip(batch_inputs_np, 0, 255)  # 确保值在[0, 255]
    batch_inputs_np = batch_inputs_np.astype(np.uint8)

    # 如果之前有从BGR转RGB，现在需要反转回去
    batch_inputs_np = batch_inputs_np[:, :, :, ::-1]  # RGB to BGR

    hsv_batch = np.zeros_like(batch_inputs_np)
    for i in range(batch_inputs.shape[0]):
        # 将每个图像转换为HSV格式
        hsv_image = cv2.cvtColor(batch_inputs_np[i], cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        # 保存原始RGB图像
        rgb_image = batch_inputs_np[i]
        # cv2.imwrite(f'rgb_image_{i}.png', rgb_image)
        # cv2.imwrite(f'h_image_{i}.png', h)
        # cv2.imwrite(f's_image_{i}.png', s)
        # cv2.imwrite(f'v_image_{i}.png', v)

        hsv_batch[i] = rgb_image

    hsv_batch = hsv_batch[:, :, :, ::-1]
    hsv_batch[:, :, :, 0] = hsv_batch[:, :, :, 0] / 179.0  # H通道归一化到[0,1]
    hsv_batch[:, :, :, 1:] = hsv_batch[:, :, :, 1:] / 255.0  # S,V通道归一化到[0,1]
    hsv_batch = torch.from_numpy(hsv_batch.copy()).permute(0, 3, 1, 2).float()
    
    return hsv_batch