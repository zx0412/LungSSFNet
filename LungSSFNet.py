import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.UnetAttention_collection.SimAM import SimAM

'''
/home/zx/results/lungMFMS_simAM_results/others/Dataset102_LUNG_abcatloss_gelu_branchSim_beforeUPsim
'''
def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,  # 输入通道数
                 dct_h, dct_w,  # DCT 高度和宽度
                 frequency_branches=16,  # 频率分支数
                 frequency_selection='top',  # 频率选择方式
                 reduction=16):  # 通道缩减比例
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]  # 确保频率分支数合法
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        # 获取频率索引
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # 初始化固定的DCT权重
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx),
                                 self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        # 定义平均池化和最大池化
        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = x

        # 如果输入尺寸不等于DCT尺寸，进行自适应平均池化
        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        # 初始化频谱特征
        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        # 平均频谱特征
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq

        # 通过全连接层计算注意力图
        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        # 合并注意力图并通过sigmoid激活
        multi_spectral_attention_map = F.sigmoid(
            multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y,
                                                                                                            mapper_y,
                                                                                                            tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)


class MFMSAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 scale_branches=3,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8,
                 groups=32):
        super(MFMSAttentionBlock, self).__init__()

        self.att3D = SimAM()
        self.scale_branches = scale_branches
        self.frequency_branches = frequency_branches
        self.block_repetition = block_repetition
        self.min_channel = min_channel
        self.min_resolution = min_resolution

        # 初始化多尺度分支
        self.multi_scale_branches = nn.ModuleList([])
        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2 ** scale_idx
            if inter_channel < self.min_channel:
                inter_channel = self.min_channel

            self.multi_scale_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1 + scale_idx,
                          dilation=1 + scale_idx, groups=groups, bias=False),
                nn.BatchNorm2d(in_channels), nn.GELU(),
                nn.Conv2d(in_channels, inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inter_channel), nn.GELU()
            ))

        # 频率到宽高的映射
        c2wh = dict([(32, 112), (64, 56), (128, 28), (256, 14), (512, 7)])
        self.multi_frequency_branches = nn.ModuleList([])
        self.multi_frequency_branches_conv1 = nn.ModuleList([])
        self.multi_frequency_branches_conv2 = nn.ModuleList([])
        self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])

        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2 ** scale_idx
            if inter_channel < self.min_channel:
                inter_channel = self.min_channel

            if frequency_branches > 0:
                self.multi_frequency_branches.append(
                    nn.Sequential(
                        MultiFrequencyChannelAttention(inter_channel, c2wh[in_channels], c2wh[in_channels],
                                                       frequency_branches, frequency_selection)))
            self.multi_frequency_branches_conv1.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Sigmoid()))
            self.multi_frequency_branches_conv2.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels), nn.GELU()))

    def forward(self, x):
        feature_aggregation = 0
        for scale_idx in range(self.scale_branches):
            # get scale features
            if scale_idx < self.scale_branches - 1:
                feature = F.avg_pool2d(x, kernel_size=2 ** scale_idx, stride=2 ** scale_idx, padding=0) if int(x.shape[2] // 2 ** scale_idx) >= self.min_resolution else x
            else:
                feature = self.att3D(x)
            feature = self.multi_scale_branches[scale_idx](feature)
            if self.frequency_branches > 0:
                feature = self.multi_frequency_branches[scale_idx](feature)
            # get SA attention
            spatial_attention_map = self.multi_frequency_branches_conv1[scale_idx](feature)
            # attention Fusion
            feature = self.multi_frequency_branches_conv2[scale_idx](
                feature * (1 - spatial_attention_map) * self.alpha_list[scale_idx] + feature * spatial_attention_map *
                self.beta_list[scale_idx])
            # attention reshape to x
            feature_aggregation += F.interpolate(feature, size=None, scale_factor=2 ** scale_idx, mode='bilinear',
                                                 align_corners=None) if (x.shape[2] != feature.shape[2]) or (
                        x.shape[3] != feature.shape[3]) else feature
        # 平均聚合特征
        feature_aggregation /= self.scale_branches
        # feature_no_param = self.att3D(x)  # 3D behind
        # feature_aggregation = (feature_aggregation + feature_no_param) / 2
        # 加上输入特征
        feature_aggregation += x

        return feature_aggregation


class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_connection_channels,
                 scale_branches=3,
                 frequency_branches=16,
                 frequency_selection='top',
                 block_repetition=1,
                 min_channel=64,
                 min_resolution=8):
        super(UpsampleBlock, self).__init__()

        self.att3D = SimAM()
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels),
                                      nn.GELU())
        self.conv4 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels),
                                      nn.GELU())

        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.beta1 = nn.Parameter(torch.tensor(1.0))

        in_channels = in_channels + skip_connection_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.GELU())

        self.attention_layer = MFMSAttentionBlock(out_channels, scale_branches, frequency_branches, frequency_selection,
                                                  block_repetition, min_channel, min_resolution)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.GELU())

    def forward(self, x, skip_connection=None):
        # nn.ConvTranspose2d 适合需要学习上采样参数的场景，能够灵活控制输出特征，但计算开销较大。
        # F.interpolate 适合不需要学习上采样参数的场景，计算效率高，使用方便，但上采样模式固定
        x1 = self.att3D(x)
        x = F.interpolate(x1, size=None, scale_factor=2, mode='bilinear', align_corners=None)
        x2_1 = torch.cat([x, skip_connection], dim=1)
        x2_2 = self.conv4(abs(self.conv3(x) - skip_connection))
        x = self.alpha1 * self.conv1(x2_1) + self.beta1 *x2_2
        # x = self.conv1(x)
        x = self.attention_layer(x)
        x = self.conv2(x)
        return x


# lung
class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.modules.instancenorm.InstanceNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True,
                                                   track_running_stats=False),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.modules.instancenorm.InstanceNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True,
                                                   track_running_stats=False),
            nn.GELU(), )
        # self.cbam = CBAM(out_ch)
        # self.attention = SimAM()

    def forward(self, x):
        # x = self.attention(self.conv(x))
        x = self.conv(x)
        # x = self.cbam(x)
        return x


class conv_pooling(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_pooling, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            nn.modules.instancenorm.InstanceNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True,
                                                   track_running_stats=False),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.modules.instancenorm.InstanceNorm2d(out_ch, eps=1e-05, momentum=0.1, affine=True,
                                                   track_running_stats=False),
            nn.GELU(), )
        # self.cbam = CBAM(out_ch)
        # self.attention = SimAM()

    def forward(self, x):
        # x = self.attention(self.conv(x))
        x = self.conv(x)
        # x = self.cbam(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, x):
        x = self.up(x)
        return x


# model
class LUNG_MFMS(nn.Module):

    def __init__(self, in_ch=3,
                 out_ch=1,
                 scale_branches=3,  # 尺度分支数量
                 frequency_branches=16,  # 频率分支数量
                 frequency_selection='top',  # 频率选择方式
                 block_repetition=1,  # 模块重复次数
                 min_channel=64,  # 最小通道数
                 min_resolution=8, ):  # 最小分辨率

        super(LUNG_MFMS, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]
        self.attention = SimAM()
        # encoder
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_pooling(filters[0], filters[1])
        self.Conv3 = conv_pooling(filters[1], filters[2])
        self.Conv4 = conv_pooling(filters[2], filters[3])
        self.Conv5 = conv_pooling(filters[3], filters[4])
        self.Conv6 = conv_pooling(filters[4], filters[4])
        self.Conv7 = conv_pooling(filters[4], filters[4])
        self.Conv8 = conv_pooling(filters[4], filters[4])

        # up and decode
        self.Up8 = up_conv(filters[4], filters[4])
        self.Up_conv8 = conv_block(filters[5], filters[4])

        self.Up7 = up_conv(filters[4], filters[4])
        self.Up_conv7 = conv_block(filters[5], filters[4])

        self.Up6 = up_conv(filters[4], filters[4])
        self.Up_conv6 = conv_block(filters[5], filters[4])

        # 解码器阶段
        self.decoder_stage1 = UpsampleBlock(filters[4], filters[3],
                                            filters[3], scale_branches, frequency_branches,
                                            frequency_selection, block_repetition, min_channel, min_resolution)
        self.decoder_stage2 = UpsampleBlock(filters[3], filters[2],
                                            filters[2], scale_branches, frequency_branches,
                                            frequency_selection, block_repetition, min_channel, min_resolution)
        self.decoder_stage3 = UpsampleBlock(filters[2], filters[1],
                                            filters[1], scale_branches, frequency_branches,
                                            frequency_selection, block_repetition, min_channel, min_resolution)
        self.decoder_stage4 = UpsampleBlock(filters[1], filters[0],
                                            filters[0], scale_branches, frequency_branches,
                                            frequency_selection, block_repetition, min_channel, min_resolution)
        # self.Up5 = up_conv(filters[4], filters[3])
        # self.Up_conv5 = conv_block(filters[4], filters[3])

        # self.Up4 = up_conv(filters[3], filters[2])
        # self.Up_conv4 = conv_block(filters[3], filters[2])

        # self.Up3 = up_conv(filters[2], filters[1])
        # self.Up_conv3 = conv_block(filters[2], filters[1])

        # self.Up2 = up_conv(filters[1], filters[0])
        # self.Up_conv2 = conv_block(filters[1], filters[0])

        # seg_head
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        if x.size()[1] == 1:
            # ([12, 1, 512, 512]) -> ([12, 3, 512, 512])
            x = x.repeat(1, 3, 1, 1)
        # encoder
        e1 = self.Conv1(x)

        e2 = self.Conv2(e1)

        e3 = self.Conv3(e2)

        e4 = self.Conv4(e3)

        e5 = self.Conv5(e4)

        e6 = self.Conv6(e5)

        e7 = self.Conv7(e6)

        e8 = self.Conv8(e7)
        # decoder
        d8 = self.Up8(e8)
        d8 = torch.cat((e7, d8), dim=1)
        d8 = self.Up_conv8(d8)

        d7 = self.Up7(d8)
        d7 = torch.cat((e6, d7), dim=1)
        d7 = self.Up_conv7(d7)

        d6 = self.Up6(d7)
        d6 = torch.cat((e5, d6), dim=1)
        d6 = self.Up_conv6(d6)

        # add multi_frequency and multi_scale
        d5 = self.decoder_stage1(d6, e4)

        # d5 = self.Up5(d6)
        # d5 = torch.cat((e4, d5), dim=1)
        # d5 = self.Up_conv5(d5)

        d4 = self.decoder_stage2(d5, e3)
        # d4 = self.Up4(d5)
        # d4 = torch.cat((e3, d4), dim=1)
        # d4 = self.Up_conv4(d4)

        d3 = self.decoder_stage3(d4, e2)
        # d3 = self.Up3(d4)
        # d3 = torch.cat((e2, d3), dim=1)
        # d3 = self.Up_conv3(d3)

        d2 = self.decoder_stage4(d3, e1)
        # d2 = self.Up2(d3)
        # d2 = torch.cat((e1, d2), dim=1)
        # d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        # invert seg outputs so that the largest segmentation prediction is returned first
        out = torch.flip(out, [1])
        return out


if __name__ == '__main__':
    a = torch.randn(4, 3, 512, 512)
    network = LUNG_MFMS(in_ch=3, out_ch=3)
    output = network(a)
    from torchinfo import summary
    summary(network, (4, 3, 512, 512))
    print(output.shape)
