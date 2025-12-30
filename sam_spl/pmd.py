import torch
import torch.nn as nn
import torch.nn.functional as F
# import pywt
import numpy as np
from functools import partial
import pywt
import pywt.data
# from timm.layers import DropPath

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class MBWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',ssm_ratio=1,forward_type="v05",):
        super(MBWTConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 如果输入通道和输出通道不一致，添加一个卷积层进行转换
        if in_channels != out_channels:
            self.channel_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        else:
            self.channel_conv = None
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, out_channels, out_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=out_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )

        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, out_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(out_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=out_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        # 如果需要调整通道数
        if self.channel_conv is not None:
            x = self.channel_conv(x)
            
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)
        return x


class Get_curvature(nn.Module):
    def __init__(self):
        super(Get_curvature, self).__init__()
        kernel_v1 = [[0, -1, 0],
                     [0, 0, 0],
                     [0, 1, 0]]
        kernel_h1 = [[0, 0, 0],
                     [-1, 0, 1],
                     [0, 0, 0]]
        kernel_h2 = [[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [1, 0, -2, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]
        kernel_v2 = [[0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, -2, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0]]
        kernel_w2 = [[1, 0, -1],
                     [0, 0, 0],
                     [-1, 0, 1]]
        kernel_h1 = torch.FloatTensor(kernel_h1).unsqueeze(0).unsqueeze(0)
        kernel_v1 = torch.FloatTensor(kernel_v1).unsqueeze(0).unsqueeze(0)
        kernel_v2 = torch.FloatTensor(kernel_v2).unsqueeze(0).unsqueeze(0)
        kernel_h2 = torch.FloatTensor(kernel_h2).unsqueeze(0).unsqueeze(0)
        kernel_w2 = torch.FloatTensor(kernel_w2).unsqueeze(0).unsqueeze(0)
        self.weight_h1 = nn.Parameter(data=kernel_h1, requires_grad=False)
        self.weight_v1 = nn.Parameter(data=kernel_v1, requires_grad=False)
        self.weight_v2 = nn.Parameter(data=kernel_v2, requires_grad=False)
        self.weight_h2 = nn.Parameter(data=kernel_h2, requires_grad=False)
        self.weight_w2 = nn.Parameter(data=kernel_w2, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v1, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h1, padding=1)
            x_i_v2 = F.conv2d(x_i.unsqueeze(1), self.weight_v2, padding=2)
            x_i_h2 = F.conv2d(x_i.unsqueeze(1), self.weight_h2, padding=2)
            x_i_w2 = F.conv2d(x_i.unsqueeze(1), self.weight_w2, padding=1)
            sum = torch.pow((torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2)), 3 / 2)
            fg = torch.mul(torch.pow(x_i_v, 2), x_i_v2) + 2 * torch.mul(torch.mul(x_i_v, x_i_h), x_i_w2) + torch.mul(
                torch.pow(x_i_h, 2), x_i_h2)
            fh = torch.mul(torch.pow(x_i_v, 2), x_i_h2) - 2 * torch.mul(torch.mul(x_i_v, x_i_h), x_i_w2) + torch.mul(
                torch.pow(x_i_h, 2), x_i_v2)
            x_i = torch.div(torch.abs(fg - fh), sum + 1e-10)
            x_i = torch.div(torch.abs(fh), sum + 1e-10)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, out_dims):
        super(FeatureEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, out_dims[0], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dims[0], out_dims[0], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(out_dims[0], out_dims[1], kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(out_dims[1], out_dims[1], kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(out_dims[1], out_dims[2], kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(out_dims[2], out_dims[2], kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(out_dims[2], out_dims[3], kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(out_dims[3], out_dims[3], kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Stage 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x1 = x

        # Stage 2
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)
        x2 = x

        # Stage 3
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool3(x)
        x3 = x

        # Stage 4
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x4 = x

        return x1, x2, x3, x4


class PMD_features(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(PMD_features, self).__init__()
        # self.PMD_head = Get_curvature()
        self.wavelet_decomp = MBWTConv2d(in_dims, out_dims, stride=2)
        self.PMD_head = Get_gradient_nopadding()
        # self.feature_ext = FeatureEncoder(out_dims)

    def forward(self, images):
        wavelet_images = self.wavelet_decomp(images)
        PMD_images = self.PMD_head(wavelet_images)
        # PMD_feature = self.feature_ext(PMD_images)

        return PMD_images

# class Adapter(nn.Module):
#     def __init__(self, out_dims):
#         super(Adapter, self).__init__()
#         self.PMD_head = Get_gradient_nopadding()
#         self.feature_ext = FeatureEncoder(out_dims)
#
#     def forward(self, images):
#         PMD_images = self.PMD_head(images)
#         PMD_feature = self.feature_ext(PMD_images)
#
#         return PMD_feature

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


if __name__ == '__main__':
    x = torch.rand(2, 64, 4, 7)
    dys = DySample(64)
    print(dys(x).shape)