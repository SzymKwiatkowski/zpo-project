import albumentations
import timm


class Augmentations(object):
    def __init__(self):
        self.__init__()

    @staticmethod
    def basic_augmentation():
        return albumentations.Compose([
            albumentations.Rotate(limit=10, p=1.0),
            albumentations.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=1.0),
            albumentations.CenterCrop(512, 512),
            albumentations.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ])

    @staticmethod
    def flip_augmentation():
        return albumentations.Compose([
            albumentations.Rotate(limit=10, p=1.0),
            albumentations.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=1.0),
            albumentations.CenterCrop(512, 512),
            albumentations.Flip(),
            albumentations.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ])

    @staticmethod
    def complicated_augmentations():
        return albumentations.Compose([
            albumentations.Rotate(limit=10, p=1.0),
            albumentations.RandomScale(),
            albumentations.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=1.0),
            albumentations.CenterCrop(512, 512),
            albumentations.Flip(),
            albumentations.ColorJitter(),
            albumentations.HorizontalFlip(),
            albumentations.ChannelDropout(),
            albumentations.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ])

    @staticmethod
    def complicated_augmentations_with_greyscale():
        return albumentations.Compose([
            albumentations.Rotate(limit=10, p=1.0),
            albumentations.RandomScale(),
            albumentations.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=1.0),
            albumentations.CenterCrop(512, 512),
            albumentations.Flip(),
            albumentations.ToGray(p=0.2),
            albumentations.ColorJitter(),
            albumentations.HorizontalFlip(),
            albumentations.ChannelDropout(),
            albumentations.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
