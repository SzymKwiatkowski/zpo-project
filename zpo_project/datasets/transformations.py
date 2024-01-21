import albumentations.pytorch
import timm


class Transformations(object):
    def __init__(self):
        self.__init__()

    @staticmethod
    def basic_transformation():
        return albumentations.Compose([
            albumentations.CenterCrop(512, 512),
            albumentations.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ])

    @staticmethod
    def deduced_transforms():
        return albumentations.Compose([
            albumentations.CenterCrop(512, 512),
            albumentations.MotionBlur(blur_limit=3, p=0.2),
            albumentations.Flip(),
            albumentations.Equalize(),
            albumentations.FancyPCA(),
            albumentations.GaussNoise(),
            albumentations.HorizontalFlip(),
            albumentations.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
