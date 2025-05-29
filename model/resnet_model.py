import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from pytorch_model_summary import summary


# resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)
# return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)

class Residual2Block(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 =  nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BottleNeck3Block(nn.Module):
    expansion = 4   # 64 to 256

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Res_default(nn.Module):

    def __init__(self, block, num_classes=1000,channel_array=[], stack_array=[], dropout_value=None, groups=1, width_per_group=64, zero_init_residual=False):
        super().__init__()
        self.num_classes = num_classes
        if dropout_value is None:
            self.is_dropout = False
        else:
            self.is_dropout = True
            self.dropout_value = dropout_value

        self.batch_norm = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group


        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.batch_norm(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.stack_layer(block, channel_array[0], stack_array[0])
        self.layer2 = self.stack_layer(block, channel_array[1], stack_array[1], stride=2, dilate=False)
        self.layer3 = self.stack_layer(block, channel_array[2], stack_array[2], stride=2, dilate=False)
        self.layer4 = self.stack_layer(block, channel_array[3], stack_array[3], stride=2, dilate=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck3Block):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, Residual2Block):
                    nn.init.constant_(m.bn2.weight, 0)




    def stack_layer(self, block, io_channels, stack_level, stride=1, dilate=False):
        # planes = stack_level = [3, 4, 6, 3]
        # block = Residual2Block or BottleNeck3Block

        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != io_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, io_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(io_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, io_channels, stride, downsample, self.groups,
                            self.base_width, previous_dilation, nn.BatchNorm2d))
        self.inplanes = io_channels * block.expansion
        for _ in range(1, stack_level):
            layers.append(block(self.inplanes, io_channels, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        # print(f"size of x in conv1:{x.size()}")
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(f"size of x in maxpool:{x.size()}")

        x = self.layer1(x)
        # print(f"size of x in layer1:{x.size()}")
        x = self.layer2(x)
        # print(f"size of x in layer2:{x.size()}")
        x = self.layer3(x)
        # print(f"size of x in layer3:{x.size()}")
        x = self.layer4(x)
        # print(f"size of x in layer4:{x.size()}")

        x = self.avgpool(x)
        # print(f"size of x in avgpool:{x.size()}")
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # print(f"size of x in fc:{x.size()}")

        return x

    def forward(self, x):
        return self._forward_impl(x)

def Res34_Net(num_classes,**kwargs):
    return Res_default(Residual2Block, num_classes,[64, 128, 256, 512], [3, 4, 6, 3], **kwargs)

def Res34_Net(num_classes,**kwargs):
    return Res_default(Residual2Block, num_classes,[64, 128, 256, 512], [3, 4, 6, 3], **kwargs)

def Res101_Net(num_classes,**kwargs):
    return Res_default(BottleNeck3Block, num_classes, [64, 128, 256, 512], [3, 4, 23, 3], **kwargs)


if __name__ == '__main__':
    # model = VGG16_NET(101, 0.2)
    # model = Custom_NET(101, 0.2)
    # model = models.resnet34()
    model = models.resnet101()
    print(summary(model, torch.zeros(1, 3, 224, 224), show_input=False, show_hierarchical=True))
    # model_2 = Res34_Net(1000)
    model_2 = Res101_Net(1000)

    print(summary(model_2, torch.zeros(1, 3, 224, 224), show_input=False, show_hierarchical=True))
