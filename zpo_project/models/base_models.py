import timm
from transformers import AutoFeatureExtractor, ResNetForImageClassification

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
    def resnet18_model_v2(embedding_size):
        return timm.create_model('resnet18.tv_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def efficientnet_b0_model(embedding_size):
        return timm.create_model('efficientnet_b0.ra_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def resnet18_model_feature_extractor(embedding_size):
        return ResNetForImageClassification.from_pretrained("microsoft/resnet-18")

    @staticmethod
    def conv_next_small_model(embedding_size):
        return timm.create_model('convnext_small.fb_in22k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def mobilenet_100_model(embedding_size):
        return timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True, num_classes=embedding_size)
