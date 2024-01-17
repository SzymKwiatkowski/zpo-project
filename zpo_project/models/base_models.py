import timm
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch

class BaseModels(object):
    def __init__(self):
        self.__init__()

    @staticmethod
    def basic_model(embedding_size):
        return timm.create_model('resnet10t', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def resnet34_model(embedding_size):
        return timm.create_model('resnet34.a1_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def resnet50_model(embedding_size):
        return timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def resnet18_model(embedding_size):
        return timm.create_model('resnet18.a1_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def efficientnet_b2(embedding_size):
        return timm.create_model('efficientnet_b2.ra_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def convnext_v2_base(embedding_size):
        return timm.create_model('convnextv2_base.fcmae', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def tf_efficientnetv2_b3(embedding_size):
        return timm.create_model('tf_efficientnetv2_b3.in21k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def convnext_pico_model(embedding_size):
        return timm.create_model('convnextv2_pico.fcmae', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def mobilenect_100_model(embedding_size):
        return timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def tf_efficientnet_b1(embedding_size):
        return timm.create_model('tf_efficientnet_b1.aa_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def tf_efficientnet_b0(embedding_size):
        return timm.create_model('efficientnet_b0.ra_in1k', pretrained=True, num_classes=embedding_size)
    
    @staticmethod
    def mobile_vit(embedding_size):
        return timm.create_model('mobilevit_s.cvnets_in1k', pretrained=True, num_classes=embedding_size)
    
