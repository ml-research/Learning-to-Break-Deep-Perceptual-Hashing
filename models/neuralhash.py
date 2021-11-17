import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import load_model, numpy_helper
import math


class Conv2dDynamicSamePadding(nn.Conv2d):
    """Taken from https://github.com/lukemelas/EfficientNet-PyTorch/blob/7e8b0d312162f335785fb5dcfa1df29a75a1783a/efficientnet_pytorch/utils.py#L215"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
                         dilation, groups, bias)
        self.stride = self.stride if len(
            self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw) 
        # Formula: https://github.com/onnx/onnx/blob/master/docs/Operators.md#averagepool
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [
                pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2
            ])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class CustomActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu6(x + 3) / 6


class NeuralHash(nn.Module):
    def __init__(self, onnx_model=None):
        super().__init__()
        self.layers = []

        # Conv0
        self.conv0 = nn.Sequential(
            Conv2dDynamicSamePadding(
                3, 16, 3, 2, dilation=1, groups=1, bias=False),
            nn.InstanceNorm2d(16, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.Hardswish()
        )
        # bottleneck1
        self.bottleneck1 = nn.Sequential(
            Conv2dDynamicSamePadding(
                16, 16, 1, 1, dilation=1, groups=1, bias=False),
            nn.InstanceNorm2d(16, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(),
            Conv2dDynamicSamePadding(
                16, 16, kernel_size=(3, 3), stride=(2, 2), groups=16, bias=False),
            nn.InstanceNorm2d(16, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
        )

        self.se1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 4, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1)),
            CustomActivation()
        )

        self.bottleneck1_2 = nn.Sequential(
            nn.ReLU(),
            Conv2dDynamicSamePadding(
                16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(16, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
        )

        # bottleneck2
        self.bottleneck2 = nn.Sequential(
            Conv2dDynamicSamePadding(
                16, 56, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(56, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(),
            Conv2dDynamicSamePadding(
                56, 56, kernel_size=(3, 3), stride=(2, 2), groups=56, bias=False),
            nn.InstanceNorm2d(56, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(),
            Conv2dDynamicSamePadding(
                56, 24, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(24, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
        )

        # bottleneck3
        self.bottleneck3 = nn.Sequential(
            Conv2dDynamicSamePadding(24, 64, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(64, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(),
            Conv2dDynamicSamePadding(64, 64, kernel_size=(
                3, 3), stride=(1, 1), groups=64, bias=False),
            nn.InstanceNorm2d(64, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.ReLU(),
            Conv2dDynamicSamePadding(64, 24, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(24, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        # bottleneck4
        self.bottleneck4 = nn.Sequential(
            Conv2dDynamicSamePadding(24, 72, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(72, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.Hardswish(),
            Conv2dDynamicSamePadding(72, 72, kernel_size=(
                5, 5), stride=(2, 2), groups=72, bias=False),
            nn.InstanceNorm2d(72, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck4_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(72, 18, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(18, 72, kernel_size=(1, 1), stride=(1, 1)),
            CustomActivation()
        )

        self.bottleneck4_2 = nn.Sequential(
            nn.Hardswish(),
            Conv2dDynamicSamePadding(72, 32, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(32, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        # Bottleneck5
        self.bottleneck5 = nn.Sequential(
            Conv2dDynamicSamePadding(32, 184, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(184, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.Hardswish(),
            Conv2dDynamicSamePadding(184, 184, kernel_size=(
                5, 5), stride=(1, 1), groups=184, bias=False),
            nn.InstanceNorm2d(184, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck5_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(184, 46, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(46, 184, kernel_size=(1, 1), stride=(1, 1)),
            CustomActivation()
        )

        self.bottleneck5_2 = nn.Sequential(
            nn.Hardswish(),
            Conv2dDynamicSamePadding(184, 32, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(32, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck6 = nn.Sequential(
            Conv2dDynamicSamePadding(32, 184, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(184, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.Hardswish(),
            Conv2dDynamicSamePadding(184, 184, kernel_size=(
                5, 5), stride=(1, 1), groups=184, bias=False),
            nn.InstanceNorm2d(184, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck6_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(184, 46, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(46, 184, kernel_size=(1, 1), stride=(1, 1)),
            CustomActivation()
        )

        self.bottleneck6_2 = nn.Sequential(
            nn.Hardswish(),
            Conv2dDynamicSamePadding(184, 32, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(32, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck7 = nn.Sequential(
            Conv2dDynamicSamePadding(32, 88, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(88, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.Hardswish(),
            Conv2dDynamicSamePadding(88, 88, kernel_size=(
                5, 5), stride=(1, 1), groups=88, bias=False),
            nn.InstanceNorm2d(88, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck7_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(88, 22, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(22, 88, kernel_size=(1, 1), stride=(1, 1)),
            CustomActivation()
        )

        self.bottleneck7_2 = nn.Sequential(
            nn.Hardswish(),
            Conv2dDynamicSamePadding(88, 40, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(40, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck8 = nn.Sequential(
            Conv2dDynamicSamePadding(40, 112, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(112, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.Hardswish(),
            Conv2dDynamicSamePadding(112, 112, kernel_size=(
                5, 5), stride=(1, 1), groups=112, bias=False),
            nn.InstanceNorm2d(112, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck8_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(112, 28, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(28, 112, kernel_size=(1, 1), stride=(1, 1)),
            CustomActivation()
        )

        self.bottleneck8_2 = nn.Sequential(
            nn.Hardswish(),
            Conv2dDynamicSamePadding(112, 40, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(40, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck9 = nn.Sequential(
            Conv2dDynamicSamePadding(40, 216, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(216, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.Hardswish(),
            Conv2dDynamicSamePadding(216, 216, kernel_size=(
                5, 5), stride=(2, 2), groups=216, bias=False),
            nn.InstanceNorm2d(216, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck9_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(216, 54, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(54, 216, kernel_size=(1, 1), stride=(1, 1)),
            CustomActivation()
        )

        self.bottleneck9_2 = nn.Sequential(
            nn.Hardswish(),
            Conv2dDynamicSamePadding(216, 72, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(72, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck10 = nn.Sequential(
            Conv2dDynamicSamePadding(72, 432, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(432, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.Hardswish(),
            Conv2dDynamicSamePadding(432, 432, kernel_size=(
                5, 5), stride=(1, 1), groups=432, bias=False),
            nn.InstanceNorm2d(432, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck10_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(432, 108, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(108, 432, kernel_size=(1, 1), stride=(1, 1)),
            CustomActivation()
        )

        self.bottleneck10_2 = nn.Sequential(
            nn.Hardswish(),
            Conv2dDynamicSamePadding(432, 72, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(72, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck11 = nn.Sequential(
            Conv2dDynamicSamePadding(72, 432, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(432, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.Hardswish(),
            Conv2dDynamicSamePadding(432, 432, kernel_size=(
                5, 5), stride=(1, 1), groups=432, bias=False),
            nn.InstanceNorm2d(432, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck11_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(432, 108, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(108, 432, kernel_size=(1, 1), stride=(1, 1)),
            CustomActivation()
        )

        self.bottleneck11_2 = nn.Sequential(
            nn.Hardswish(),
            Conv2dDynamicSamePadding(432, 72, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(72, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False)
        )

        self.bottleneck12 = nn.Sequential(
            Conv2dDynamicSamePadding(72, 432, kernel_size=(
                1, 1), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(432, eps=9.999999974752427e-07,
                              momentum=0.1, affine=True, track_running_stats=False),
            nn.Hardswish()
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dDynamicSamePadding(
                432, 1280, kernel_size=(1, 1), stride=(1, 1)),
            nn.Hardswish(),
            nn.Conv2d(1280, 500, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(500, 128, kernel_size=(1, 1), stride=(1, 1))
        )

        if onnx_model:
            self.copy_weights(onnx_model)

    def copy_weights(self, onnx_model):
        onnx_weights = []
        for w in onnx_model.graph.initializer:
            weight = torch.from_numpy(numpy_helper.to_array(w))
            if weight.shape == torch.Size([1]):
                continue
            onnx_weights.append(weight)

        for outer_module in self.children():
            for layer in outer_module:
                if hasattr(layer, 'weight'):
                    if layer.weight.data.shape == onnx_weights[0].shape:
                        layer.weight.data = onnx_weights.pop(0)
                    else:
                        raise RuntimeError(
                            f'Incompatible weight matrix for layer {layer}.')
                if hasattr(layer, 'bias'):
                    if layer.bias is None:
                        continue
                    if layer.bias.data.shape == onnx_weights[0].shape:
                        layer.bias.data = onnx_weights.pop(0)
                    else:
                        raise RuntimeError(
                            f'Incompatible bias matrix for layer {layer}.')
        if len(onnx_weights) > 0:
            print(f'{len(onnx_weights)} parameters not assigned')
        else:
            print('All parameters assigned')

    def forward(self, input):
        # Conv0
        x = self.conv0(input)
        # Bottleneck1
        x = self.bottleneck1(x)
        x = x * self.se1(x)
        x = self.bottleneck1_2(x)
        # Bottleneck2
        x = self.bottleneck2(x)
        # Bottleneck3
        x = x + self.bottleneck3(x)
        # Bottleneck4
        x = self.bottleneck4(x)
        x = x * self.bottleneck4_se(x)
        x = self.bottleneck4_2(x)
        # Bottleneck5
        x_shortcut = x
        x = self.bottleneck5(x)
        x = x * self.bottleneck5_se(x)
        x = self.bottleneck5_2(x)
        x = x + x_shortcut
        # Bottleneck6
        x_shortcut = x
        x = self.bottleneck6(x)
        x = x * self.bottleneck6_se(x)
        x = self.bottleneck6_2(x)
        x = x + x_shortcut
        # Bottleneck7
        x = self.bottleneck7(x)
        x = x * self.bottleneck7_se(x)
        x = self.bottleneck7_2(x)
        # Bottleneck8
        x_shortcut = x
        x = self.bottleneck8(x)
        x = x * self.bottleneck8_se(x)
        x = self.bottleneck8_2(x)
        x = x + x_shortcut
        # Bottleneck9
        x = self.bottleneck9(x)
        x = x * self.bottleneck9_se(x)
        x = self.bottleneck9_2(x)
        # Bottleneck10
        x_shortcut = x
        x = self.bottleneck10(x)
        x = x * self.bottleneck10_se(x)
        x = self.bottleneck10_2(x)
        x = x + x_shortcut
        # Bottleneck11
        x_shortcut = x
        x = self.bottleneck11(x)
        x = x * self.bottleneck11_se(x)
        x = self.bottleneck11_2(x)
        x = x + x_shortcut
        # Bottleneck12
        x = self.bottleneck12(x)
        # Final layers
        x = self.output(x)
        return x
