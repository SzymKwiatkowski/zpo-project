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
    def inception_v3_model(embedding_size):
        return timm.create_model('inception_v3.gluon_in1k', pretrained=True, num_classes=embedding_size)
    
    @staticmethod
    def xception41p_random_augument_model(embedding_size):
        return timm.create_model('xception41p.ra3_in1k', pretrained=True, num_classes=embedding_size)
    
    @staticmethod
    def xception65_random_augument_model(embedding_size):
        return timm.create_model('xception65.ra3_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def resnet101_random_augument_model(embedding_size):
        return timm.create_model('resnet101d.ra2_in1k', pretrained=True, num_classes=embedding_size)
    
    @staticmethod
    def resnet101_a1_recipe_model(embedding_size):
        return timm.create_model('resnet101.a1h_in1k', pretrained=True, num_classes=embedding_size)
    
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
    def darknet53(embedding_size):
        return timm.create_model('darknet53.c2ns_in1k', pretrained=True, num_classes=embedding_size)
    
    @staticmethod
    def ghostnetv2_100(embedding_size):
        return timm.create_model('ghostnetv2_100.in1k', pretrained=True, num_classes=embedding_size)
    
    @staticmethod
    def mixnet_l(embedding_size):
        return timm.create_model('mixnet_l.ft_in1k', pretrained=True, num_classes=embedding_size)
    
    @staticmethod
    def fbnetv3_b(embedding_size):
        return timm.create_model('fbnetv3_b.ra2_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def tf_efficientnetv2_b3(embedding_size):
        return timm.create_model('tf_efficientnetv2_b3.in21k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def convnext_pico_model(embedding_size):
        return timm.create_model('convnextv2_pico.fcmae', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def mobilenet_100_model(embedding_size):
        return timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def mobilenetv3_rw_model(embedding_size):
        return timm.create_model('mobilenetv3_rw.rmsp_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def tf_efficientnet_b1(embedding_size):
        return timm.create_model('tf_efficientnet_b1.aa_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def tf_efficientnet_b0(embedding_size):
        return timm.create_model('efficientnet_b0.ra_in1k', pretrained=True, num_classes=embedding_size)
    
    @staticmethod
    def mobile_vit(embedding_size):
        return timm.create_model('mobilevit_s.cvnets_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def mixnet(embedding_size):
        return timm.create_model('mixnet_l', pretrained=True, num_classes=embedding_size)
        
    @staticmethod
    def levit(embedding_size):
        return timm.create_model('levit_128.fb_dist_in1k', pretrained=True, num_classes=embedding_size)
    
    @staticmethod
    def davit(embedding_size):
        return timm.create_model('davit_base.msft_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def convmixer_model(embedding_size):
        return timm.create_model('convmixer_768_32.in1k', pretrained=True, num_classes=embedding_size)
    
    @staticmethod
    def dm_nfnet_f0(embedding_size):
        return timm.create_model('dm_nfnet_f0.dm_in1k', pretrained=True, num_classes=embedding_size)
    
    @staticmethod
    def resnetaa101d_model(embedding_size):
        return timm.create_model('resnetaa101d.sw_in12k', pretrained=True, num_classes=embedding_size)
    
    @staticmethod
    def tresnet_m_model(embedding_size):
        return timm.create_model('tresnet_m.miil_in21k', pretrained=True, num_classes=embedding_size)
