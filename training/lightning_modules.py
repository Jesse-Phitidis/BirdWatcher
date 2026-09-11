import pytorch_lightning as pl
import torch
from torch import nn
import timm
from torchmetrics import Accuracy, F1Score, Precision, Recall
from pathlib import Path
import pandas as pd

class ClassifierLightningModule(pl.LightningModule):

    def __init__(
            self, 
            model_name: str, 
            num_classes: int,
            loss_fn: nn.Module,
            class_mapping_csv_path: str,
            ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.class_mapping_csv_path = Path(class_mapping_csv_path)
        self.class_mapping = pd.read_csv(self.class_mapping_csv_path)

        self.model = timm.create_model(
            self.model_name, 
            pretrained=True, 
            num_classes=self.num_classes,
            cache_dir=str(Path(__file__).parents[1] / "models")
            )

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes, average="none")
        self.f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="none")
        self.precision = Precision(task="multiclass", num_classes=self.num_classes, average="none")
        self.recall = Recall(task="multiclass", num_classes=self.num_classes, average="none")
        self.preds, self.targets = [], []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        pred = self(batch["img"])
        loss = self.loss_fn(pred, batch["class_id"])
        self.log(name="loss", value=loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        pred = self(batch["img"])
        self.preds.append(pred)
        self.targets.append(batch["class_id"])

    def on_validation_epoch_end(self):
        full_preds = torch.cat(self.preds)
        full_targets = torch.cat(self.targets)
        loss = self.loss_fn(full_preds, full_targets)
        accuracy = self.accuracy(full_preds, full_targets)
        f1 = self.f1(full_preds, full_targets)
        precision = self.precision(full_preds, full_targets)
        recall = self.recall(full_preds, full_targets)

        self.log(name="val_loss", value=loss, on_step=False, on_epoch=True)
        self.log(name="val_accuracy", value=accuracy.mean(), on_step=False, on_epoch=True)
        self.log(name="val_f1", value=f1.mean(), on_step=False, on_epoch=True)
        self.log(name="val_precision", value=precision.mean(), on_step=False, on_epoch=True)
        self.log(name="val_recall", value=recall.mean(), on_step=False, on_epoch=True)

        for i in range(self.num_classes):
            class_name = self.class_mapping.loc[i, "label"]
            self.log(name=f"val_accuracy/{class_name}", value=accuracy[i], on_step=False, on_epoch=True)
            self.log(name=f"val_f1/{class_name}", value=f1[i], on_step=False, on_epoch=True)
            self.log(name=f"val_precision/{class_name}", value=precision[i], on_step=False, on_epoch=True)
            self.log(name=f"val_recall/{class_name}", value=recall[i], on_step=False, on_epoch=True)

        self.preds.clear()
        self.targets.clear()
