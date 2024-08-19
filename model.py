import torch
import torch.nn as nn
import torch.nn.functional as F

from feature import CQTWithShift
from backbone import OctaveNet


class LKEModel(nn.Module):
    def __init__(
        self,

        feature_args={},
        model_args={},

    ) -> None:
        super().__init__()

        self.feature_layer = CQTWithShift(**feature_args)
        self.backbone = OctaveNet(**model_args)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def train_step(self, batch):
        x, y = batch["x"], batch["y"]
        x, y = self.feature_layer(x, y, shift=True)

        y_pred = self.backbone(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred

    def val_step(self, batch):
        x, y = batch["x"], batch["y"]
        x, y = self.feature_layer(x, y, shift=False)

        y_pred = self.backbone(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred

    def pred_step(self, batch):
        x, y = batch["x"], batch["y"]
        x, y = self.feature_layer(x, y, shift=False)

        return self.backbone(x)
