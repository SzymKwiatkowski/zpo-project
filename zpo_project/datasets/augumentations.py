import albumentations
import timm


class Augumentations(object):
    def __init__(self):
        self.__init__()

    @staticmethod
    def basic_augumentation():
        return albumentations.Compose([
            albumentations.Rotate(limit=10, p=1.0),
            albumentations.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=1.0),
            albumentations.CenterCrop(512, 512),
            albumentations.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
