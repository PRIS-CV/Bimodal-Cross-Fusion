import torch
from torch import nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmengine.model import BaseModel
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image


# 定义解码器
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.embed_dim = embed_dim

        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output,pos_encoding1,pos_encoding2,h,w, src_mask=None, tgt_mask=None):
        """
         Args:
            x: 解码器输入，形状为 (bs, H*W, c)
            encoder_output: 编码器输出，形状为 (bs, H'*W', c)
            src_mask: 编码器输入的掩码，形状为 (H'*W', H'*W')
            tgt_mask: 解码器输入的掩码，形状为 (H*W, H*W)
        Returns:
            解码器输出，形状为 (bs, H*W, c)
        """
        batch_size, seq_len, _ = x.shape

        # 添加位置编码到自注意力的Q和K
        x_with_pos = x + pos_encoding1  # (bs,H*W,c)

        # 自注意力机制
        x_with_pos = x_with_pos.transpose(0, 1)  # (H*W, bs, c)
        x = x.transpose(0, 1)  # (H*W, bs, c)
        self_attn_output, self_attn_weights = self.self_attention(
            query=x_with_pos,
            key=x_with_pos,
            value=x,
            attn_mask=tgt_mask
        )
        self_attn_output = self_attn_output.transpose(0, 1)  # (bs, H*W, c)
        x = x.transpose(0, 1)  # (bs, H*W, c)
        x = x + self.dropout1(self_attn_output)
        x = self.norm1(x)
        x_enhance = x
        # self_attn_cam = self_attn_weights.mean(dim=-1)
        # self_attn_cam = self_attn_cam.view(batch_size,h,w)
        # self_attn_cam = self_attn_cam[0].to("cuda")
        # generate_cam_with_original_image(
        #     "rgb_image_0.png",
        #     self_attn_cam,
        #     save_path="self_attn_cam_0.png"
        # )

        # 自注意力输出作为交叉注意力的Q
        cross_attn_query = x + pos_encoding1  # (bs, H*W, c)

        # 添加位置编码到交叉注意力的K
        encoder_output_with_pos = encoder_output + pos_encoding2  # (bs, H'*W', c)

        # 交叉注意力机制
        cross_attn_query = cross_attn_query.transpose(0, 1)  # (H*W, bs, c)
        encoder_output_with_pos = encoder_output_with_pos.transpose(0, 1)  # (H'*W', bs, c)
        encoder_output = encoder_output.transpose(0, 1)  # (H'*W', bs, c)
        cross_attn_output, cross_attn_weights = self.cross_attention(
            query=cross_attn_query,
            key=encoder_output_with_pos,
            value=encoder_output,
            attn_mask=src_mask
        )
        cross_attn_output = cross_attn_output.transpose(0, 1)  # (bs, H*W, c)
        x = x + self.dropout2(cross_attn_output)
        x = self.norm2(x)

        # cross_attn_cam = cross_attn_weights.mean(dim=-1)
        # cross_attn_cam = cross_attn_cam.view(batch_size,h,w)
        # cross_attn_cam = cross_attn_cam[0].to("cuda")
        # generate_cam_with_original_image(
        #     "rgb_image_0.png",
        #     cross_attn_cam,
        #     save_path="cross_attn_cam_0.png"
        # )

        # 前馈神经网络
        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)
        x = self.norm3(x)

        return x_enhance,x

def generate_cam_with_original_image(original_image_path, cam_weights, save_path=None):
    original_image = Image.open(original_image_path).convert("RGB")
    original_image_size = original_image.size
    transform = transforms.ToTensor()
    original_image_tensor = transform(original_image).unsqueeze(0)
    if len(cam_weights.shape) == 2:
        cam_weights = cam_weights.unsqueeze(0)
    cam_weights = F.interpolate(
        cam_weights.unsqueeze(0),  # (1, 1, H, W)
        size=(original_image_size[1], original_image_size[0]),  # (H, W)
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    cam_weights = (cam_weights - cam_weights.min()) / (cam_weights.max() - cam_weights.min())
    cam_weights_np = cam_weights.squeeze(0).detach().cpu().numpy()  # (H, W)
    cam_heatmap = plt.get_cmap('jet')(cam_weights_np)[:, :, :3]  # (H, W, 3) 彩色热力图
    cam_heatmap = torch.from_numpy(cam_heatmap).permute(2, 0, 1).float()  # (3, H, W)
    # original_image_tensor = original_image_tensor.to('cuda')
    overlayed_image = original_image_tensor.squeeze(0) * 0.5 + cam_heatmap * 0.5  # (3, H, W)
    overlayed_image = torch.clamp(overlayed_image, 0, 1)
    if save_path is not None:
        save_image(overlayed_image, save_path)

def visualize_feature_map(feature_map, title,save_path):
    feature_map = feature_map.detach().cpu().numpy()
    plt.imshow(feature_map[0, 0, :, :], cmap='viridis')
    plt.title(title)
    plt.colorbar()
    # plt.savefig(save_path)
    plt.show()
    plt.close()


# 定义BCF(Bimodal Cross-Attention Feature Fusion)模块
@MODELS.register_module()
class BCF(BaseModel):
    def __init__(self,bs,embed_dim,num_heads,hidden_dim,dropout):
        super(BCF,self).__init__()
        # 定义解码器堆栈
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim,num_heads,hidden_dim,dropout) for _ in range(1)
        ])
        # 定义自注意力模块
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 定义conv1x1
        self.conv1x1 = nn.Conv2d(embed_dim*2, embed_dim, kernel_size=1, stride=1, padding=0)
        # 定义可学习位置编码
        self.pos_encoding1 = nn.Parameter(torch.zeros(bs, 2000, embed_dim))  # 用于自注意力Q/K和交叉注意力Q
        self.pos_encoding2 = nn.Parameter(torch.zeros(bs, 2000, embed_dim))  # 用于交叉注意力K

    def forward(self,f_rgb,f_hsv):
        bs,c,h,w = f_rgb.shape
        seq_len = h*w
        # 调整位置编码
        if seq_len > self.pos_encoding1.size(1):
            pos_encoding1 = torch.nn.functional.interpolate(
                self.pos_encoding1.permute(0, 2, 1),  # [1, c, max_seq_len]
                size=seq_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # [1, seq_len, c]
            pos_encoding2 = torch.nn.functional.interpolate(
                self.pos_encoding2.permute(0, 2, 1),  # [1, c, max_seq_len]
                size=seq_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # [1, seq_len, c]
        else:
            pos_encoding1 = self.pos_encoding1[:, :seq_len, :]
            pos_encoding2 = self.pos_encoding2[:, :seq_len, :]

        pos_encoding1 = pos_encoding1[ : bs, : , : ]
        pos_encoding2 = pos_encoding2[ : bs, : , : ]

        # 铺平特征(bs,c,h,w)->(bs,h*w,c)
        f_rgb = f_rgb.view(bs,c,-1).permute(0,2,1)
        f_hsv = f_hsv.view(bs,c,-1).permute(0,2,1)
        # 经过多层解码器
        f_rgb_enhanced = f_rgb
        f_fuse = f_hsv
        for decoder_layer in self.decoder_layers:
            f_rgb_enhanced,f_fuse = decoder_layer(f_rgb_enhanced,f_fuse,pos_encoding1,pos_encoding2,h,w)
        f_fuse_with_pos = f_fuse + pos_encoding1
        f_fuse_with_pos = f_fuse_with_pos.transpose(0, 1)
        f_fuse = f_fuse.transpose(0, 1)
        # 经过自注意力模块
        f_fuse_self,_ = self.self_attention(
            query=f_fuse_with_pos,
            key=f_fuse_with_pos,
            value=f_fuse,
            attn_mask=None
        )
        f_fuse_self = f_fuse_self.transpose(0, 1)  # (bs, H*W, c)
        f_fuse = f_fuse.transpose(0, 1)  # (bs, H*W, c)
        f_fuse = f_fuse + self.dropout(f_fuse_self)
        f_fuse = self.norm(f_fuse)

        # 变换为原本形状(bs,h*w,c)->(bs,c,h,w)
        f_rgb = f_rgb.view(bs, h, w, c).permute(0, 3, 1, 2)
        f_hsv = f_hsv.view(bs,h,w,c).permute(0,3,1,2)
        f_rgb_enhanced = f_rgb_enhanced.view(bs, h, w, c).permute(0, 3, 1, 2)
        f_fuse = f_fuse.view(bs, h, w, c).permute(0, 3, 1, 2)
        # 连接，经过conv1x1
        f_fuse = torch.cat((f_rgb,f_fuse),dim=1)
        f_fuse = self.conv1x1(f_fuse)

        return f_fuse

