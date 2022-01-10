from .resnet_pretrained import ResNet, _resnet, Bottleneck
import torch
import random

def mixup_data(x, y, lam):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    index = index.to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class ResNetMixup(torch.nn.Module):
    def __init__(self, model: ResNet):
        super().__init__()
        self.model = model
    
    def forward(self, x, target=None, mixup=False, mixup_hidden=True, lam=0.4, emb_out_layer=-1):
        if target is not None:  # using manifold mixup pretraining strategy
            if mixup_hidden:
                layer_mix = random.randint(0, 4)
            elif mixup:
                layer_mix = 0
            else:
                layer_mix = None   
            out = x
            target_a = target_b = target
            if layer_mix == 0:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)
            out = self.model.conv1(out)
            out = self.model.bn1(out)
            out = self.model.relu(out)
            out = self.model.maxpool(out)
            out = self.model.layer1(out)
            if layer_mix == 1:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)
            out = self.model.layer2(out)
            if layer_mix == 2:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)
            out = self.model.layer3(out)
            if layer_mix == 3:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)
            out = self.model.layer4(out)
            if layer_mix == 4:
                out, target_a, target_b, lam = mixup_data(out, target, lam=lam)
            out = self.model.avgpool(out)
            out = torch.flatten(out, 1)
            return out, target_a, target_b
        elif emb_out_layer == -1:  # no mixup, the default embedding output layer is the final layer (-1)
            out = self.model(x)
            return out
        else:  # then emb_out_layer should be in [1, 2, 3]
            result = []
            out = self.model.conv1(x)
            out = self.model.bn1(out)
            out = self.model.relu(out)
            out = self.model.maxpool(out)
            out = self.model.layer1(out)
            # [I][MyProto] out embedding of layer1 .shape: torch.Size([5, 256, 7, 7])
            # [I][MyProto] out embedding of layer2 .shape: torch.Size([5, 512, 4, 4])
            # [I][MyProto] out embedding of layer3 .shape: torch.Size([5, 1024, 2, 2])
            # [I][MyProto] out embedding of layer4 .shape: torch.Size([5, 2048, 1, 1])
            if emb_out_layer == 0:
                out = torch.mean(out, [2, 3])
                return out  # (num_of_samples, 256)
            if emb_out_layer == -2:
                result.append(torch.mean(out, [2,3]))
            out = self.model.layer2(out)
            if emb_out_layer == 1:
                out = torch.mean(out, [2, 3])
                return out  # (num_of_samples, 512)
            if emb_out_layer == -2:
                result.append(torch.mean(out, [2,3]))
            out = self.model.layer3(out)
            if emb_out_layer == 2:
                out = torch.mean(out, [2, 3])
                return out  # (num_of_samples, 1024)
            if emb_out_layer == -2:
                result.append(torch.mean(out, [2,3]))
            out = self.model.layer4(out)
            out = self.model.avgpool(out)
            out = torch.flatten(out, 1)
            if emb_out_layer == -2:
                result.append(out)
                return result
            return out

from torch import nn
from .utils import load_state_dict_from_url

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2MixWrap(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class MobileNetV2Mix(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2Mix, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        self.block_number = [-1, 3, 6, 10, 13, 16, 18]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.ModuleList(features)
        # self.features = nn.Sequential(*features)

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes),
        # )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x, target=None, mixup=False, mixup_hidden=True, lam=0.4, emb_out_layer=-1):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass

        if target is not None:
            if mixup_hidden:
                layer_mix = random.randint(0, len(self.block_number) - 1)
            elif mixup:
                layer_mix = 0
            else:
                layer_mix = None
            target_a = target_b = target
            if layer_mix == 0:
                x, target_a, target_b, lam = mixup_data(x, target, lam=lam)
            for i, feature in enumerate(self.features):
                x = feature(x)
                if layer_mix is not None and self.block_number[layer_mix] == i:
                    x, target_a, target_b, lam = mixup_data(x, target, lam=lam)
            x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
            return x, target_a, target_b

        if emb_out_layer == -1:
            emb_out_layer = len(self.block_number) - 2

        out_list = []        
        for i, feature in enumerate(self.features):
            x = feature(x)
            if i == self.block_number[emb_out_layer + 1] and emb_out_layer >= 0:
                return nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
            if emb_out_layer == -2 and i in self.block_number:
                out_list.append(nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1))
        assert emb_out_layer == -2
        return out_list

    def forward(self, x, target=None, mixup=False, mixup_hidden=True, lam=0.4, emb_out_layer=-1):
        return self._forward_impl(x, target, mixup, mixup_hidden, lam, emb_out_layer)


def mobilenetmix(pretrained=True, progress=True, state_dict_path=None,**kwargs):
    model = MobileNetV2Mix(**kwargs)
    if pretrained:
        if state_dict_path is None:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        else:
            state_dict = torch.load(state_dict_path)
        state_dict = {k: state_dict[k] for k  in state_dict if 'classifier' not in k}
        model.load_state_dict(state_dict)
    model = MobileNetV2MixWrap(model)
    return model

def resnetmix50(pretrained=True, progress=True, state_dict_path=None, **kwargs):
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, state_dict_path,
                   **kwargs)
    return ResNetMixup(model)

def wide_resnet50_2_mix(pretrained=True, progress=True, state_dict_path=None, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    model = _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, state_dict_path,
                   **kwargs)
    return ResNetMixup(model)

def resnet152_mix(pretrained=True, progress=True, state_dict_path=None, **kwargs):
    model = _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, state_dict_path,
                   **kwargs)
    return ResNetMixup(model)