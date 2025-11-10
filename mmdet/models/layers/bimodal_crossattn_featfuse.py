# import torch
# from torch import nn
# import torch.nn.functional as F
# from mmdet.registry import MODELS
# from mmengine.model import BaseModel
#
# # 定义单头交叉注意力模块
# class CrossAttention(nn.Module):
#     def __init__(self,C,kernel_size=1):
#         super(CrossAttention,self).__init__()
#         self.C = C
#         self.kernel_size = kernel_size
#         if isinstance(kernel_size, int):
#             padding = (kernel_size - 1) // 2
#         else:
#             padding = ((kernel_size[0] - 1) // 2,(kernel_size[1] - 1) // 2)
#         # 定义Q、K、V卷积层
#         self.query_conv = nn.Conv2d(C,C,kernel_size=kernel_size,padding=padding)
#         self.key_conv = nn.Conv2d(C,C,kernel_size=kernel_size,padding=padding)
#         self.value_conv = nn.Conv2d(C,C,kernel_size=kernel_size,padding=padding)
#         # 添加层归一化
#         self.norm1 = nn.LayerNorm(C)
#         self.norm2 = nn.LayerNorm(C)
#
#     def forward(self,feat1,feat2):
#         # 获取形状
#         B,C,H,W = feat1.shape
#
#         # # 保存残差连接
#         # identity = feat1
#
#         # 生成Q、K、V
#         Q = self.query_conv(feat2).view(B, C, H*W)
#         K = self.key_conv(feat1).view(B, C, H*W)
#         V = self.value_conv(feat1).view(B, C, H*W)
#
#         # 添加层归一化
#         Q = self.norm1(Q.transpose(1, 2)).transpose(1, 2)
#         K = self.norm1(K.transpose(1, 2)).transpose(1, 2)
#         V = self.norm2(V.transpose(1, 2)).transpose(1, 2)
#
#
#         # 计算注意力权重矩阵
#         A = torch.matmul(Q.transpose(-2,-1),K)
#         A = A / (C ** 0.5)
#         A = F.softmax(A,dim=-1)
#
#
#         # 加权求和
#         out = torch.matmul(V,A.transpose(-2,-1))
#         out = out.view(B,C,H,W)
#
#         # # 残差连接
#         # out = out + identity
#
#         return out
#
# # 定义自注意力机制模块
# class SelfAttention(nn.Module):
#     def __init__(self,C):
#         super(SelfAttention,self).__init__()
#         self.C = C
#         self.scale = C ** -0.5
#
#         # Q,K,V投影层
#         self.q_proj = nn.Conv2d(C, C, kernel_size=1)
#         self.k_proj = nn.Conv2d(C, C, kernel_size=1)
#         self.v_proj = nn.Conv2d(C, C, kernel_size=1)
#
#         # 输出投影层
#         self.out_proj = nn.Conv2d(C, C, kernel_size=1)
#
#         # 归一化层
#         self.norm1 = nn.LayerNorm(C)
#         self.norm2 = nn.LayerNorm(C)
#
#         # Dropout
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self,x):
#         B,C,H,W = x.shape
#
#          # 保存残差连接
#         # identity = x
#
#         # 生成Q,K,V
#         Q = self.q_proj(x).view(B, C, -1)  # B,C,HW
#         K = self.k_proj(x).view(B, C, -1)  # B,C,HW
#         V = self.v_proj(x).view(B, C, -1)  # B,C,HW
#
#         # 应用归一化
#         Q = self.norm1(Q.transpose(1, 2)).transpose(1, 2)
#         K = self.norm1(K.transpose(1, 2)).transpose(1, 2)
#         V = self.norm2(V.transpose(1, 2)).transpose(1, 2)
#
#         # 计算注意力
#         attn = torch.matmul(Q.transpose(-2, -1), K) * self.scale  # B,HW,HW
#         attn = F.softmax(attn, dim=-1)
#         attn = self.dropout(attn)
#
#         # 注意力加权
#         out = torch.matmul(V, attn.transpose(-2, -1))  # B,C,HW
#         out = out.view(B, C, H, W)
#
#         # 输出投影
#         out = self.out_proj(out)
#
#         # 残差连接
#         # out = out + identity
#
#         return out
#
# # 定义FFN模块
# class FFN(nn.Module):
#     def __init__(self,C):
#         super(FFN,self).__init__()
#         self.fc1 = nn.Linear(C,C*2)
#         self.activation = nn.GELU()  # 完全非线性
#         self.fc2 = nn.Linear(C*2,C)
#         self.norm = nn.LayerNorm(C)
#
#         # # 定义dropout，防止过拟合
#         # self.dropout = nn.Dropout(0.1)
#
#          # 添加残差连接
#         self.use_residual = False
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
#         # 保存残差连接
#         if self.use_residual:
#             identity = x
#         # 应用全连接层
#         x = self.fc1(x)
#         x = self.activation(x)
#         # x = self.dropout(x)
#         x = self.fc2(x)
#         # 残差连接
#         if self.use_residual:
#             x = x + identity
#         # 层归一化
#         x = self.norm(x)
#         # 转换回 [B, C, H, W]
#         x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
#
#         return x
#
# # 定义BCF(Bimodal Cross-Attention Feature Fusion)模块
# @MODELS.register_module()
# class BCF(BaseModel):
#     def __init__(self,C,kernel_size_list):
#         super(BCF,self).__init__()
#         # 定义特征通道数
#         self.C = C
#         # 定义不同核大小交叉注意力机制列表
#         CA_RGB_list = []
#         CA_HSV_list = []
#         for k in kernel_size_list:
#             CA_RGB_list.append(CrossAttention(C=C, kernel_size=k))
#             CA_HSV_list.append(CrossAttention(C=C, kernel_size=k))
#         self.CA_RGB = nn.ModuleList(CA_RGB_list)
#         self.CA_HSV = nn.ModuleList(CA_HSV_list)
#         # 定义不同conv1x1，渐进式通道调整
#         n = len(kernel_size_list)
#         self.conv1x1_rgb1 = nn.Conv2d(C * n, C//2, kernel_size=1)
#         self.conv1x1_hsv1 = nn.Conv2d(C * n, C//2, kernel_size=1)
#         self.conv1x1_rgb2 = nn.Conv2d(C//2,C//8,kernel_size=1)
#         self.conv1x1_hsv2 = nn.Conv2d(C//2,C//8,kernel_size=1)
#         self.conv1x1_fuse = nn.Conv2d(C*2+C//8*2,C,kernel_size=1)
#         # 定义FFN
#         self.FFN_rgb = FFN(C//8)
#         self.FFN_hsv = FFN(C//8)
#         # 添加层归一化
#         self.norm1 = nn.LayerNorm(C//8)
#         self.norm2 = nn.LayerNorm(C//8)
#         # 添加最终的归一化层
#         self.final_norm = nn.LayerNorm(C)
#         # 定义自注意力机制
#         self.SA_rgb = SelfAttention(C//2)
#         self.SA_hsv = SelfAttention(C//2)
#
#     def forward(self,f_rgb,f_hsv):
#         # f_rgb: (B,C,H,W)
#         # 进行交叉注意力操作，并将加权后的特征存储
#         f_rgb_ = []
#         f_hsv_ = []
#         for i in range(0,len(self.CA_RGB)):
#             f_rgb_.append(self.CA_RGB[i](f_rgb,f_hsv))
#             f_hsv_.append(self.CA_HSV[i](f_hsv,f_rgb))
#
#         # 堆叠特征，conv1x1调整通道数
#         f_rgb_ = self.conv1x1_rgb1(torch.cat(f_rgb_,dim=1))  # (B,C*len(k_list),H,W) -> (B,256,H,W)
#         f_hsv_ = self.conv1x1_hsv1(torch.cat(f_hsv_,dim=1))
#
#         # 自注意力处理
#         f_rgb_ = self.SA_rgb(f_rgb_)  # (B,256,H,W) -> (B,256,H,W)
#         f_hsv_ = self.SA_hsv(f_hsv_)
#
#         # 调整通道
#         f_rgb_ = self.conv1x1_rgb2(f_rgb_)  # (B,256,H,W) -> (B,64,H,W)
#         f_hsv_ = self.conv1x1_hsv2(f_hsv_)
#
#         # 经过FFN模块，进一步细化全局信息
#         f_rgb_ = self.FFN_rgb(f_rgb_)  # (B,64,H,W) -> (B,64,H,W)
#         f_hsv_ = self.FFN_hsv(f_hsv_)
#
#         # 添加层归一化
#         B, C, H, W = f_rgb_.shape
#         f_rgb_ = self.norm1(f_rgb_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         f_hsv_ = self.norm2(f_hsv_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         # 按RGB->HSV->RGB_->HSV_顺序连接
#         f_fuse_ = torch.cat([f_rgb,f_hsv,f_rgb_,f_hsv_],dim=1)   # (B,C*2+128,H,W)
#
#         # conv1x1调整通道数
#         f_fuse = self.conv1x1_fuse(f_fuse_)  # (B,C*2+128,H,W) -> (B,C,H,W)
#
#         # 最终的归一化
#         f_fuse = self.final_norm(f_fuse.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         return f_fuse
#
