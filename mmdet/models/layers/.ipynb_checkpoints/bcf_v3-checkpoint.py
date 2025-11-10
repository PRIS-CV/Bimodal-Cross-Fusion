# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmengine.model import BaseModel
# from mmdet.registry import MODELS
# import matplotlib.pyplot as plt



# class GlobalAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(GlobalAttention, self).__init__()
#         self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
#         self.softmax = nn.Softmax(dim=2)

#     def forward(self, x):
#         batch_size, _, height, width = x.size()
#         attention = self.conv(x).view(batch_size, 1, -1)  # [batch_size, 1, height*width]
#         attention = self.softmax(attention)
#         attention = attention.view(batch_size, 1, height, width)
#         return x * attention

# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction_ratio, in_channels)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
#         max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
#         out = avg_out + max_out
#         out = self.sigmoid(out).unsqueeze(2).unsqueeze(3)
#         return x * out

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.conv(out)
#         out = self.sigmoid(out)
#         return x * out
    
# class CombinedAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
#         super(CombinedAttention, self).__init__()
#         self.global_attention1 = GlobalAttention(in_channels)
#         self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
#         self.spatial_attention = SpatialAttention(kernel_size)
#         self.global_attention2 = GlobalAttention(in_channels)

#     def forward(self, x):
#         x = self.global_attention1(x)
#         x = self.channel_attention(x)
#         x = self.spatial_attention(x)
#         x = self.global_attention2(x)
#         return x

# class AdaptiveWeightedFusion(nn.Module):
#     def __init__(self, in_channels):
#         super(AdaptiveWeightedFusion, self).__init__()
#         self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x_rgb, x_hsv):
#         x_cat = torch.cat([x_rgb, x_hsv], dim=1)
#         weights = self.sigmoid(self.conv(x_cat))
#         x_fuse = x_rgb * weights + x_hsv * (1 - weights)
#         return x_fuse



# def visualize_feature_map(feature_map, title,save_path):
#     feature_map = feature_map.detach().cpu().numpy()
#     plt.imshow(feature_map[0, 0, :, :], cmap='viridis')
#     plt.title(title)
#     plt.colorbar()
#     # plt.savefig(save_path)
#     plt.show()
#     plt.close()


# @MODELS.register_module()
# class BCF(BaseModel):
#     def __init__(self,in_channels):
#         super(BCF,self).__init__()
#         self.hsv_enhance = CombinedAttention(in_channels=in_channels)
#         # self.weight = nn.Parameter(torch.ones(1, in_channels, 1, 1))
#         self.fuse = AdaptiveWeightedFusion(in_channels=in_channels)
    
#     def forward(self,x_rgb,x_hsv):
#         # visualize_feature_map(x_rgb, "RGB Feature Map1","Fused_Feature_Map1.png")
#         # visualize_feature_map(x_hsv, "HSV Feature Map","Fused_Feature_Map1.png")
#         x_hsv = self.hsv_enhance(x_hsv)
#         # visualize_feature_map(x_hsv, "HSV Feature Map1","Fused_Feature_Map1.png")
#         # x_fuse = x_rgb+x_hsv*self.weight
#         # print(self.weight)
#         x_fuse = self.fuse(x_rgb,x_hsv)
#         # visualize_feature_map(x_fuse, "Fused Feature Map1","Fused_Feature_Map1.png")
#         return x_fuse