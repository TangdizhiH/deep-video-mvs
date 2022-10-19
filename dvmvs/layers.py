import torch


def down_conv_layer(input_channels, output_channels, kernel_size):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=1,
            bias=False),
        torch.nn.BatchNorm2d(output_channels),
        torch.nn.ReLU(),
        torch.nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=2,
            bias=False),
        torch.nn.BatchNorm2d(output_channels),
        torch.nn.ReLU())


def up_conv_layer(input_channels, output_channels, kernel_size):
    return torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False),
        torch.nn.BatchNorm2d(output_channels),
        torch.nn.ReLU())


def conv_layer(input_channels, output_channels, kernel_size, stride, apply_bn_relu):
    # yang: wrapper for unified convolutiona layer with or without batchnorm and relu. You can learn this to write some standardized layers for different purposes.
    if apply_bn_relu:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                bias=False),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True))
    else:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                bias=False))


def depth_layer_3x3(input_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(input_channels, 1, 3, padding=1),
        torch.nn.Sigmoid())
