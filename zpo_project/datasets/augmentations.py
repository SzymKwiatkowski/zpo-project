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
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ])

    @staticmethod
    def minimal_augmentation():
        return albumentations.Compose([
            albumentations.Rotate(limit=10, p=0.85),
            albumentations.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=1.0),
            albumentations.RandomScale(),
            albumentations.CenterCrop(512, 512),
            albumentations.GaussianBlur(),
            albumentations.PixelDropout(),
            albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
            albumentations.FancyPCA(alpha=0.1, always_apply=False, p=0.5),
            albumentations.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ])

    @staticmethod
    def complicated_augmentations():
        return albumentations.Compose([
            albumentations.Rotate(limit=15, p=1.0),
            # albumentations.RandomScale(),
            albumentations.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=1.0),
            albumentations.CenterCrop(512, 512),
            albumentations.Flip(),
            albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            # albumentations.FancyPCA(alpha=0.1, always_apply=False, p=0.5),  
            albumentations.HorizontalFlip(p=0.5),
            albumentations.GaussianBlur(),
            # albumentations.GaussNoise(),
            albumentations.augmentations.dropout.grid_dropout.GridDropout(),
            albumentations.RandomGridShuffle(),
            albumentations.PixelDropout(),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
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
            albumentations.ColorJitter(),
            albumentations.GaussNoise(var_limit=(10, 50), p=1.0), 
            albumentations.FancyPCA(alpha=0.1, always_apply=False, p=0.5),           
            albumentations.HorizontalFlip(),
            albumentations.ChannelDropout(),
            albumentations.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ])

    @staticmethod
    def complicated_augmentations_with_greyscale_extra():
        return albumentations.Compose([
            albumentations.RandomResizedCrop(512, 512, scale=(0.8, 1.0)),
            albumentations.Rotate(limit=20, p=0.5),
            albumentations.CenterCrop(512, 512),
            albumentations.RandomScale(scale_limit=0.2, p=0.5),
            albumentations.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=0, p=0.5),
            albumentations.HorizontalFlip(p=0.7),
            albumentations.VerticalFlip(p=0.7),
            albumentations.ToGray(p=0.2),
            albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            albumentations.Perspective(scale=(0.05, 0.1), p=0.5),
            # albumentations.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            albumentations.ChannelDropout(),
            albumentations.FancyPCA(),
            albumentations.GaussNoise(),
            albumentations.GaussianBlur(),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            albumentations.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
