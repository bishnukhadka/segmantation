from modeling.backbone import resnet, xception, drn, mobilenet
"""
TODO: implement capability of the code, to add pre-trained backbone from ImageNet
"""


def build_backbone(backbone, output_stride, BatchNorm, pretrained=False):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, pretrained)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm, pretrained)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, pretrained)
    else:
        raise NotImplementedError
