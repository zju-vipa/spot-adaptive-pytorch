from typing import Dict

import torch

from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2, wrn_28_10
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .vggv2 import vgg13_bn as vgg13_bn_imagenet
from .vggv2 import vgg8_bn as vgg8_bn_imagenet
from .vggv2 import vgg11_bn as vgg11_bn_imagenet
from .vggv2 import vgg16_bn as vgg16_bn_imagenet
from .vggv2 import vgg19_bn as vgg19_bn_imagenet
from .mobilenetv2 import mobilev2
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
import torch.nn as nn




MODEL_DICT = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_28_10': wrn_28_10,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'vgg8-imagenet': vgg8_bn_imagenet,
    'vgg11-imagenet': vgg11_bn_imagenet,
    'vgg13-imagenet': vgg13_bn_imagenet,
    'vgg16-imagenet': vgg16_bn_imagenet,
    'vgg19-imagenet': vgg19_bn_imagenet,
    'MobileNetV2': mobilev2,
    'ShuffleNetV1': ShuffleV1,
    'ShuffleNetV2': ShuffleV2,
}


def get_model(model_name: str, num_classes: int, state_dict: Dict[str, torch.Tensor] = None, **kwargs):
    fn = MODEL_DICT[model_name]
    model = fn(num_classes=num_classes, **kwargs)

    if state_dict is not None:
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model



