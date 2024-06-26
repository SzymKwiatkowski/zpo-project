from pathlib import Path
from typing import Optional

from lightning import pytorch as pl
from torch.utils.data import DataLoader

from zpo_project.datasets.augmentations import Augmentations
from zpo_project.datasets.dataset_splits import DatasetSplits
from zpo_project.datasets.evaluation import EvaluationDataset
from zpo_project.datasets.metric_learning import MetricLearningDataset
from zpo_project.datasets.prediction import PredictionDataset
from zpo_project.datasets.transformations import Transformations


class MetricLearningDataModule(pl.LightningDataModule):
    def __init__(self, data_path: Path, number_of_places_per_batch: int, number_of_images_per_place: int,
                 number_of_batches_per_epoch: int, augment: bool, validation_batch_size: int, number_of_workers: int,
                 train_size: float, augmentation_selection: str = "basic_augmentation", transformations_selection: str = "basic_transformation"):
        super().__init__()

        self._data_path = Path(data_path)
        self._number_of_places_per_batch = number_of_places_per_batch
        self._number_of_images_per_place = number_of_images_per_place
        self._number_of_batches_per_epoch = number_of_batches_per_epoch
        self._validation_batch_size = validation_batch_size
        self._number_of_workers = number_of_workers
        self.train_size = train_size

        self.save_hyperparameters(ignore=['data_path', 'number_of_workers'])

        selected_transformation = getattr(Transformations, transformations_selection)
        self._transforms = selected_transformation()
        self._predict_transforms = Transformations.basic_transformation()
        selected_augmentation = getattr(Augmentations, augmentation_selection)
        self._augmentations = selected_augmentation() if augment else self._transforms

        self.train_dataset = None
        self.val_dataset = None
        self.easy_test_dataset = None
        self.medium_test_dataset = None
        self.hard_test_dataset = None
        self.predict_dataset = None

    def get_places_dirs(self, data_dir: Path) -> list[Path]:
        return sorted(
            [place_dir for place_dir in data_dir.iterdir()
             if place_dir.is_dir() and len(list(place_dir.iterdir())) >= self._number_of_images_per_place]
        )

    def get_number_of_places(self, subset: str) -> int:
        assert subset in ['train', 'val', 'test']
        return len(self.get_places_dirs(self._data_path / subset))

    def setup(self, stage: Optional[str] = None):
        train_places_dirs = self.get_places_dirs(self._data_path / 'train')
        # TODO: validation dataset size can be changed
        train_places_dirs, val_places_dirs = DatasetSplits.basic_split(train_places_dirs, self.train_size)

        print(f'Number of train places: {len(train_places_dirs)}')
        print(f'Number of val places: {len(val_places_dirs)}')

        self.train_dataset = MetricLearningDataset(
            train_places_dirs,
            self._number_of_places_per_batch,
            self._number_of_images_per_place,
            self._number_of_batches_per_epoch,
            self._augmentations,
        )
        self.val_dataset = EvaluationDataset(
            val_places_dirs,
            self._number_of_images_per_place,
            self._transforms,
        )
        self.predict_dataset = PredictionDataset(
            self._data_path / 'test',
            self._predict_transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=1, num_workers=self._number_of_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self._validation_batch_size, num_workers=self._number_of_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self._validation_batch_size, num_workers=self._number_of_workers,
        )
