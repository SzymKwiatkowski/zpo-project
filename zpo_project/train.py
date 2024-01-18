import pickle
from pathlib import Path
import argparse
import yaml

import lightning.pytorch as pl

from datamodules.metric_learning import MetricLearningDataModule
from models.model import EmbeddingModel

def train(args):
    config_file = args.config
    max_epochs = args.epochs
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    token = config['config']['NEPTUNE_API_TOKEN']
    logger = pl.loggers.NeptuneLogger(
        project='szymkwiatkowski/zpo-project',
        api_token=token)

    pl.seed_everything(42, workers=True)
    patience = 10

    # TODO: experiment with data module and model settings
    datamodule = MetricLearningDataModule(
        data_path=Path('data'),
        number_of_places_per_batch=32,
        number_of_images_per_place=4,
        number_of_batches_per_epoch=100,
        augment=True,
        validation_batch_size=32,
        number_of_workers=8,
        train_size=0.7,
        augmentation_selection="complicated_augmentations"  # Name of augmentation function from Augmentations class
    )
    model = EmbeddingModel(
        embedding_size=1024,
        lr=2e-4,
        lr_patience=10,
        lr_factor=0.5,
        model='mobilenect_100_model',
        miner="multi_similarity",
        loss_function="circle_loss",
        distance="cosine",
        distance_p=2,
        distance_power=1,
        distance_normalize_embedding=True,
        distance_is_inverted=False
    )
    model.hparams.update(datamodule.hparams)

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=-1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch}-{val_precision_at_1:.5f}', mode='max',
                                                       monitor='val_precision_at_1', verbose=True, save_last=True)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_precision_at_1', mode='max', patience=patience)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[model_summary_callback, checkpoint_callback, early_stop_callback, lr_monitor],
        accelerator='gpu',
        max_epochs=max_epochs
    )

    trainer.fit(model=model, datamodule=datamodule)
    predictions = trainer.predict(model=model, ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

    results = {}
    for prediction in predictions:
        for embedding, identifier in zip(*prediction):
            results[identifier] = embedding.tolist()

    with open('results.pickle', 'wb') as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-c', '--config', action='store', default='config.yaml')
    parser.add_argument('-e', '--epochs', action='store', default=50,
                        type=int, help='Specified number of maximum epochs')
    args = parser.parse_args()
    train(args)
