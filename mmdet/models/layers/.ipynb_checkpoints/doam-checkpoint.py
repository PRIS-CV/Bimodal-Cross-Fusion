import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
from torch.autograd import Variable
from PIL import Image
from mmdet.registry import MODELS
from mmengine.model import BaseModel


def init_weights(net, init_type='normal', gain=0.02):
    from torch.nn import init
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


@MODELS.register_module()
class DOAM(BaseModel):
    def __init__(self):
        super(DOAM, self).__init__()
        kernel_const_hori = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype='float32')
        kernel_const_hori = torch.FloatTensor(kernel_const_hori).unsqueeze(0)
        self.weight_const_hori = nn.Parameter(data=kernel_const_hori, requires_grad=False)

        self.weight_hori = self.weight_const_hori
        self.gamma = nn.Parameter(torch.zeros(1))

        kernel_const_vertical = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype='float32')
        kernel_const_vertical = torch.FloatTensor(kernel_const_vertical).unsqueeze(0)
        self.weight_const_vertical = nn.Parameter(data=kernel_const_vertical, requires_grad=False)

        self.weight_vertical = self.weight_const_vertical

        self.sigmoid = torch.nn.Sequential()
        self.sigmoid.add_module('Sigmoid',nn.Sigmoid())

        self.conv2d_1_1 = torch.nn.Sequential()
        self.conv2d_1_1.add_module('conv2d_1_1', nn.Conv2d(9, 1, kernel_size=3, padding=1))

        self.AdaptiveAverPool_5 = torch.nn.Sequential()
        self.AdaptiveAverPool_5.add_module('AdaptiveAverPool_5',nn.AdaptiveAvgPool2d((60,60)))

        self.AdaptiveAverPool_10 = torch.nn.Sequential()
        self.AdaptiveAverPool_10.add_module('AdaptiveAverPool_10', nn.AdaptiveAvgPool2d((30, 30)))

        self.AdaptiveAverPool_15 = torch.nn.Sequential()
        self.AdaptiveAverPool_15.add_module('AdaptiveAverPool_15', nn.AdaptiveAvgPool2d((20, 20)))


        # 边缘图的5个卷积
        self.conv2d_1_attention = torch.nn.Sequential()
        self.conv2d_1_attention.add_module('conv2d_1_attention',nn.Conv2d(1, 8, kernel_size=3, padding=1))
        self.conv2d_2_attention = torch.nn.Sequential()
        self.conv2d_2_attention.add_module('conv2d_2_attention', nn.Conv2d(8, 16, kernel_size=3, padding=1))
        self.conv2d_3_attention = torch.nn.Sequential()
        self.conv2d_3_attention.add_module('conv2d_3_attention', nn.Conv2d(16, 16, kernel_size=3, padding=1))
        self.conv2d_4_attention = torch.nn.Sequential()
        self.conv2d_4_attention.add_module('conv2d_4_attention', nn.Conv2d(16, 8, kernel_size=3, padding=1))
        self.conv2d_5_attention = torch.nn.Sequential()
        self.conv2d_5_attention.add_module('conv2d_5_attention', nn.Conv2d(8, 1, kernel_size=3, padding=1))
        #self.conv2d_2_attention = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        #rgb convolution
        self.conv2d_1_rgb_attention = torch.nn.Sequential()
        self.conv2d_1_rgb_attention.add_module('conv2d_1_rgb_attention', nn.Conv2d(4, 8, kernel_size=3, padding=1))
        self.conv2d_2_rgb_attention = torch.nn.Sequential()
        self.conv2d_2_rgb_attention.add_module('conv2d_2_rgb_attention', nn.Conv2d(8, 16, kernel_size=3, padding=1))
        self.conv2d_3_rgb_attention = torch.nn.Sequential()
        self.conv2d_3_rgb_attention.add_module('conv2d_3_rgb_attention', nn.Conv2d(16, 32, kernel_size=3, padding=1))
        self.conv2d_4_rgb_attention = torch.nn.Sequential()
        self.conv2d_4_rgb_attention.add_module('conv2d_4_rgb_attention', nn.Conv2d(32, 16, kernel_size=3, padding=1))
        self.conv2d_5_rgb_attention = torch.nn.Sequential()
        self.conv2d_5_rgb_attention.add_module('conv2d_5_rgb_attention', nn.Conv2d(16, 8, kernel_size=3, padding=1))

        self.GatedConv2dWithActivation = GatedConv2dWithActivation(in_channels=(8 * 3),
                                                                   out_channels=8, kernel_size=3,
                                                                   stride=1, padding=1,activation=None)

        self.conv2d_1_rgb_red_concat_5 = torch.nn.Sequential()
        self.conv2d_1_rgb_red_concat_5.add_module('conv2d_1_rgb_red_concat_5', nn.Conv2d(16, 8, kernel_size=3, padding=1))
        self.conv2d_1_rgb_red_concat_10 = torch.nn.Sequential()
        self.conv2d_1_rgb_red_concat_10.add_module('conv2d_1_rgb_red_concat_10',
                                                  nn.Conv2d(16, 8, kernel_size=3, padding=1))
        self.conv2d_1_rgb_red_concat_15 = torch.nn.Sequential()
        self.conv2d_1_rgb_red_concat_15.add_module('conv2d_1_rgb_red_concat_15',
                                                  nn.Conv2d(16, 8, kernel_size=3, padding=1))
    def RIA(self,x):
        x_shape = x.shape
        og_x = x
        refine_30 = []
        x_pooled_upsample_5  = torch.zeros((x.shape[0], 8, 300, 300), device=x.device)
        x_pooled_upsample_10 = torch.zeros((x.shape[0], 8, 300, 300), device=x.device)
        x_pooled_upsample_15 = torch.zeros((x.shape[0], 8, 300, 300), device=x.device)

        x_pooled_5 = self.AdaptiveAverPool_5(x)
        x_pooled_10 = self.AdaptiveAverPool_10(x)
        x_pooled_15 = self.AdaptiveAverPool_15(x)
        for i in range(60):
            for j in range(60):
                x_pooled_upsample_5[:,:,i*5:(i+1)*5,j*5:(j+1)*5] = x_pooled_5[:,:,i,j].unsqueeze(-1).unsqueeze(-1)

        for i in range(30):
            for j in range(30):
                x_pooled_upsample_10[:,:,i*10:(i+1)*10,j*10:(j+1)*10] = x_pooled_10[:,:,i,j].unsqueeze(-1).unsqueeze(-1)

        for i in range(20):
            for j in range(20):
                x_pooled_upsample_15[:,:,i*15:(i+1)*15,j*15:(j+1)*15] = x_pooled_15[:,:,i,j].unsqueeze(-1).unsqueeze(-1)

        x_concat_5 = torch.cat((x, x_pooled_upsample_5), 1)
        x_concat_10 = torch.cat((x, x_pooled_upsample_10), 1)
        x_concat_15 = torch.cat((x, x_pooled_upsample_15), 1)

        x_concat_5_out = self.conv2d_1_rgb_red_concat_5(x_concat_5)
        x_concat_10_out = self.conv2d_1_rgb_red_concat_10(x_concat_10)
        x_concat_15_out = self.conv2d_1_rgb_red_concat_15(x_concat_15)


        x_gated_conv_input = torch.cat((x_concat_5_out,x_concat_10_out),1)
        x_gated_conv_input = torch.cat((x_gated_conv_input, x_concat_15_out), 1)

        x_gated_conv_output = self.GatedConv2dWithActivation(x_gated_conv_input)
        return x_gated_conv_output

    def MaxMinNormalization(self,x):
        """[0,1] normaliaztion"""
        x = (x - x.min()) / (x.max() - x.min()) *255
        #x = x.astype(int)
        return x

    def forward(self,im):
        #global count
        # 调整输入形状
        original_shape = im.shape  # [bs, c, H, W]
        im = F.interpolate(im, size=(300, 300), mode='bilinear', align_corners=False)
        og_im = im  # 保存原始图像
        #im = Variable(im[:, 0].unsqueeze(1))
        x3 = Variable(im[:, 2].unsqueeze(1))  # 提取第三通道，并增加一个维度作为批处理维度
        # 将水平和垂直方向的权重转换为Variable
        weight_hori = Variable(self.weight_hori)
        weight_vertical = Variable(self.weight_vertical)

        # try:
        x_hori = F.conv2d(x3, weight_hori, padding=1 )  # 检测水平边缘
        #x_hori = self.conv2d_hori(x3)
        # except:
        #     print('horizon error')
        try:
            x_vertical = F.conv2d(x3, weight_vertical, padding=1)  # 检测垂直边缘
            #x_vertical = self.conv2d_vertical(x3)
        except:
            print('vertical error')

        #get edge image
        edge_detect = (torch.add(x_hori.pow(2),x_vertical.pow(2))).pow(0.5)  # 计算边缘检测结果


        # 对边缘检测结果进行连续卷积操作，增强边缘特征
        edge_detect_conved = self.conv2d_1_attention(edge_detect)
        edge_detect_conved = self.conv2d_2_attention(edge_detect_conved)
        edge_detect_conved = self.conv2d_3_attention(edge_detect_conved)
        edge_detect_conved = self.conv2d_4_attention(edge_detect_conved)
        edge_detect_conved = self.conv2d_5_attention(edge_detect_conved)

        # 归一化处理
        edge_detect = ((edge_detect - (edge_detect.min())) / ((edge_detect.max()) - (edge_detect.min())))

        # 原始与边缘结果合并
        rgb_red = torch.cat((im, edge_detect * 255), 1)

        # 连续卷积操作
        rgb_conved = self.conv2d_1_rgb_attention(rgb_red)
        rgb_conved = self.conv2d_2_rgb_attention(rgb_conved)
        rgb_conved = self.conv2d_3_rgb_attention(rgb_conved)
        rgb_conved = self.conv2d_4_rgb_attention(rgb_conved)
        rgb_conved = self.conv2d_5_rgb_attention(rgb_conved)

        # 应用注意力机制，细化特征图
        rgb_conved = self.RIA(rgb_conved)


        # 材料与边缘信息合并
        rgb_red_conved = torch.cat((rgb_conved, edge_detect_conved), 1)
        # 合并结果进行卷积操作
        rgb_red_conved = self.conv2d_1_1(rgb_red_conved)
        #softmax_output = self.softmax(edge_detect)
        #print('softmax_output:',softmax_output)
        #rgb_red = ((rgb_red - (rgb_red.min())) / ((rgb_red.max()) - (rgb_red.min())))

        # 应用sigmoid函数，将特征图转换为概率图
        sigmoid_output = self.sigmoid(rgb_red_conved)
        #count = count + 1

        # 门控机制，结合原始图像和预测的边缘概率图，加权融合特征
        rgb_red = self.gamma * (sigmoid_output * im) + (1 - self.gamma)*im

        # 将输出图像恢复为原始形状 [bs, c, H, W]
        rgb_red = F.interpolate(rgb_red, size=(original_shape[2], original_shape[3]), mode='bilinear', align_corners=False)

        return rgb_red#,edge_detect
