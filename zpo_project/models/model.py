import torch.linalg
from lightning import pytorch as pl
from pytorch_metric_learning import miners, losses, distances, reducers
from torchmetrics import MetricCollection

from zpo_project.metrics.multi import MultiMetric
from zpo_project.models.base_models import BaseModels


class EmbeddingModel(pl.LightningModule):
    def __init__(self,
                 embedding_size: int,
                 lr: float,
                 lr_patience: int,
                 lr_factor: float,
                 model: str = 'resnet50_model',
                 miner: str = "triplet_margin_miner",
                 loss_function: str = "triplet_loss",
                 distance: str = "euclidean",
                 distance_p: int = 2,
                 distance_power: int = 2,
                 distance_normalize_embedding: bool = True,
                 distance_is_inverted: bool = False):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        model = getattr(BaseModels, model)
        self.network = model(embedding_size)

        # TODO: The distance, the miner and the loss function are subject to change
        # TODO: Adding embedding regularization is probably a good idea
        # Selection of distance function
        if distance == "lp":
            self.distance = distances.LpDistance(
                p=distance_p,
                power=distance_power,
                normalize_embeddings=distance_normalize_embedding,
                is_inverted=distance_is_inverted
            )
        elif distance == "cosine":
            self.distance = distances.CosineSimilarity(
                # p=distance_p,
                # power=distance_power
            )
        elif distance == "dot_product":
            self.distance = distances.DotProductSimilarity(
                p=distance_p,
                power=distance_power,
                normalize_embeddings=distance_normalize_embedding,
                is_inverted=distance_is_inverted
            )
        elif distance == "snr":
            self.distance = distances.SNRDistance(
                p=distance_p,
                power=distance_power,
                normalize_embeddings=distance_normalize_embedding,
                is_inverted=distance_is_inverted
            )
        else:
            self.distance = distances.LpDistance(
                p=distance_p,
                power=distance_power,
                normalize_embeddings=distance_normalize_embedding,
                is_inverted=distance_is_inverted
            )  # Euclidean distance

        # Selectio of miner
        if miner == "multi_similarity":
            self.miner = miners.MultiSimilarityMiner(epsilon=0.15, distance=self.distance)
        elif miner == "triplet_margin_miner":
            self.miner = miners.TripletMarginMiner(distance=self.distance)
        elif miner == "hdc":
            self.miner = miners.HDCMiner(distance=self.distance)
        elif miner == "distance_weighted_miner":
            self.miner = miners.DistanceWeightedMiner(distance=self.distance)
        elif miner == "angular":
            self.miner = miners.AngularMiner(distance=self.distance)
        else:
            self.miner = miners.MultiSimilarityMiner(distance=self.distance)

        # reducer = reducers.

        # Selection of loss function
        if loss_function == "triplet_loss":
            self.loss_function = losses.TripletMarginLoss(distance=self.distance)
        elif loss_function == "tuplet_margin":
            self.loss_function = losses.TupletMarginLoss(distance=self.distance)
        elif loss_function == "nca":
            self.loss_function = losses.NCALoss(distance=self.distance)
        elif loss_function == "contrastive":
            self.loss_function = losses.ContrastiveLoss(pos_margin=0, neg_margin=1, distance=self.distance)
        elif loss_function == "pnp":
            self.loss_function = losses.PNPLoss(distance=self.distance)
        elif loss_function == "angular":
            self.loss_function = losses.AngularLoss(alpha=40, distance=self.distance)
        elif loss_function == "circle_loss":
            self.loss_function = losses.CircleLoss(distance=self.distance)
        else:
            self.loss_function = losses.TripletMarginLoss(distance=self.distance)

        self.val_outputs = None

        metrics = MetricCollection(MultiMetric(distance=self.distance))
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_pred = self.forward(x)
        loss = self.loss_function(y_pred, y, self.miner(y_pred, y))
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_pred = self.forward(x)
        self.val_outputs['preds'].append(y_pred.cpu())
        self.val_outputs['targets'].append(y.cpu())

    def predict_step(self, batch, batch_idx, **kwargs) -> tuple[torch.Tensor, list[str]]:
        x, y = batch
        x = x.squeeze(0)
        y_pred = self.forward(x)
        return y_pred.cpu(), y

    def on_validation_epoch_start(self) -> None:
        self.val_outputs = {
            'preds': [],
            'targets': [],
        }

    def on_validation_epoch_end(self) -> None:
        preds = torch.cat(self.val_outputs['preds'], dim=0)
        targets = torch.cat(self.val_outputs['targets'], dim=0)
        self.log_dict(self.val_metrics(preds, targets), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01, amsgrad=True)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.lr_patience,
        #                                                        factor=self.lr_factor)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.934)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 70, 100], gamma=0.2)  # , patience=self.lr_patience)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_precision_at_1',
        }
