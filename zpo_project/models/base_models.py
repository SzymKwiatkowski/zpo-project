import timm


class BaseModels(object):
    def __init__(self):
        self.__init__()

    @staticmethod
    def basic_model(embedding_size):
        return timm.create_model('resnet10t', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def resnet34_model(embedding_size):
        return timm.create_model('resnet34', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def resnet50_model(embedding_size):
        return timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def resnet18_model(embedding_size):
        return timm.create_model('resnet18.a1_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def resnet18_model_v2(embedding_size):
        return timm.create_model('resnet18.fb_swsl_ig1b_ft_in1k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def convnext_small_model(embedding_size):
        return timm.create_model('convnext_small.fb_in22k', pretrained=True, num_classes=embedding_size)

    @staticmethod
    def mobilenect_100_model(embedding_size):
        return timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True, num_classes=embedding_size)
