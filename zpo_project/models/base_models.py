import timm


class BaseModels(object):
    def __init__(self):
        self.__init__()

    @staticmethod
    def basic_model(embedding_size):
        return timm.create_model('resnet10t', pretrained=True, num_classes=embedding_size)
