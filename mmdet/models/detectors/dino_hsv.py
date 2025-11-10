# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union, List

import torch
from torch import Tensor, nn
from torch.nn.init import normal_
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import OptConfigType
from ..layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from ..layers import CdnQueryGeneratorv2, CdnQueryGeneratorv2HaveEmbedding
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
# from .deformable_detr import DeformableDETR, MultiScaleDeformableAttentionV2
# from .deformable_detr import DeformableDETR, MultiScaleDeformableAttentionHSV
from .dinov2_2 import DINOv2


# dinov2 通过在def forward_transformer（）中self.label_embeddings=self.dn_query_generator.label_embedding取消了xavier初始化


@MODELS.register_module()
class DINO_HSV(DINOv2):
    r"""Implementation of `DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator. Defaults to `None`.
    """

    def __init__(self, *args, dn_cfg: OptConfigType = None, **kwargs) -> None:
        super().__init__(*args, dn_cfg=dn_cfg, **kwargs)
        # assert self.as_two_stage, 'as_two_stage must be True for DINO'
        # assert self.with_box_refine, 'with_box_refine must be True for DINO'

        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)
        # self.label_embeddings = self.dn_query_generator.label_embedding

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
        # for m in self.modules():
        #     if isinstance(m, MultiScaleDeformableAttentionV2):
        #         m.init_weights()
        # for m in self.modules():
        #     if isinstance(m, MultiScaleDeformableAttentionHSV):
        #         m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        # nn.init.xavier_uniform_(self.label_embeddings.weight)
        normal_(self.level_embed)
        
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(batch_inputs,
                                                    img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats = self.extract_feat(batch_inputs)
        # TODO 判断是否要进行cam可视化
        if self.cam == True:
            import torch
            import matplotlib.pyplot as plt
            from PIL import Image
            import os

            # 假设您的特征层数据存储在变量 img_feats 中
            # img_feats[0] 的形状为 torch.Size([1, 256, 21, 40])

            # 假设 batch_data_samples 是一个包含样本信息的列表
            # 我们取第一个样本的原始尺寸和图像路径
            ori_shape = batch_data_samples[0].ori_shape  # (448, 860)
            img_path = batch_data_samples[0].img_path
            imagename = os.path.basename(img_path)
            # 创建保存图像的文件夹
            save_folder = r"D:\Projects\DINO_mmdet3\mmdetection\tools\cam/"  # 替换为您的文件夹路径
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # 保存原始图片
            original_img = Image.open(img_path)
            original_img.save(os.path.join(save_folder, imagename))

            # 处理特征层并保存热力图
            feature_layer = img_feats[0]  # 取出特征层

            for i in range(feature_layer.shape[1]):
                # 提取单个特征图
                feature_map = feature_layer[0, i, :, :]

                # 双线性插值调整特征图大小
                feature_map = torch.nn.functional.interpolate(feature_map.unsqueeze(0).unsqueeze(0),
                                                              size=ori_shape,
                                                              mode='bilinear',
                                                              align_corners=False)
                feature_map = feature_map.squeeze(0).squeeze(0)

                # 归一化特征图
                feature_min = feature_map.min()
                feature_max = feature_map.max()
                feature_map = (feature_map - feature_min) / (feature_max - feature_min)

                # 将特征图转换为numpy数组以便可视化
                feature_map_np = feature_map.cpu().numpy()

                # 创建热力图
                plt.imshow(original_img)
                plt.imshow(feature_map_np, cmap='jet', alpha=0.5)  # alpha 控制透明度
                # plt.imshow(feature_map_np, cmap='viridis', alpha=0.5)  # alpha 控制透明度
                # plt.imshow(feature_map_np, cmap='plasma', alpha=0.5)  # alpha 控制透明度

                plt.axis('off')

                # 保存热力图
                plt.savefig(os.path.join(save_folder, f"{imagename}_{i}.png"), bbox_inches='tight', pad_inches=0)
                plt.close()
            feature_map_all_channel = torch.sum(feature_layer, dim=1).unsqueeze(1)
            # 双线性插值调整特征图大小
            feature_map_all_channel = torch.nn.functional.interpolate(feature_map_all_channel,
                                                                      size=ori_shape,
                                                                      mode='bilinear',
                                                                      align_corners=False)
            feature_map_all_channel = feature_map_all_channel.squeeze(0).squeeze(0)

            # 归一化特征图
            feature_min = feature_map_all_channel.min()
            feature_max = feature_map_all_channel.max()
            feature_map_all_channel = (feature_map_all_channel - feature_min) / (feature_max - feature_min)

            # 将特征图转换为numpy数组以便可视化
            feature_map_all_channel_np = feature_map_all_channel.cpu().numpy()

            # 创建热力图
            plt.imshow(original_img)
            plt.imshow(feature_map_all_channel_np, cmap='jet', alpha=0.5)  # alpha 控制透明度
            # plt.imshow(feature_map_all_channel_np, cmap='viridis', alpha=0.5)  # alpha 控制透明度
            # plt.imshow(feature_map_all_channel_np, cmap='plasma', alpha=0.5)  # alpha 控制透明度

            plt.axis('off')

            # 保存热力图
            plt.savefig(os.path.join(save_folder, f"{imagename}_all.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

        head_inputs_dict = self.forward_transformer(batch_inputs, img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(batch_inputs=batch_inputs,
                                                    img_feats=img_feats,
                                                    batch_data_samples=batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def forward_transformer(
            self,
            batch_inputs: Tensor,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        self.label_embeddings = self.dn_query_generator.label_embedding
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            batch_inputs=batch_inputs, mlvl_feats=img_feats, batch_data_samples=batch_data_samples)

        # encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict,
                                                    batch_data_samples=batch_data_samples,
                                                    visualization=self.visualization_sampling_point)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        # decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict) # decoder输出包含6层decoder的输出feat，和reference points7个代表是预测结果，但是LFT还需要叠加一次下一层预测偏移量所以这里需要保存下来备用

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict,
                                                    batch_data_samples=batch_data_samples,
                                                    visualization=self.visualization_sampling_point)  # decoder输出包含6层decoder的输出feat，和reference points7个代表是预测结果，但是LFT还需要叠加一次下一层预测偏移量所以这里需要保存下来备用

        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_transformer(
            self,
            batch_inputs: Tensor,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask'.
        """
        batch_size = mlvl_feats[0].size(0)

        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        input_img_h, input_img_w = batch_input_shape
        masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        # 原始图片[(179, 320), (315, 307)]需要规范到输入尺寸(315, 320)，扩充后的那部分无效信息需要进行掩码，这里其实就是将边缘的无效部分进行掩码，比如图片下方的几行或者右边的几行

        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))
        # 将掩码和位置编码进行双线性插值，将其从输入尺寸reshape到多层特征尺寸。 这里的position encoding就是最简单的三角函数位置编码，编码内容与位置和通道数有关而跟内容无关

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1,
                                                                   -1)  # 这里的level_embed就是deformable transformer中的scale-level embedding
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            mask = mask.flatten(1)
            spatial_shape = (h, w)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        mask_flatten = torch.cat(mask_flatten, 1)
        # 将特征、位置编码、掩码全部摊平成 bs，num_feat_points, dim 的形式

        spatial_shapes = torch.as_tensor(  # (num_level, 2)
            spatial_shapes,
            dtype=torch.long,
            device=feat_flatten.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(  # (bs, num_level, 2)
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        # 计算出有效特征的比例，在每个level中
        encoder_inputs_dict = dict(
            batch_inputs=batch_inputs,
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, batch_inputs: Tensor,
                        feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        **kwargs  # TODO 加入此接口
                        ) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            batch_inputs=batch_inputs,
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs
        )
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)
        return encoder_outputs_dict


