import torch.nn as nn

from feature import CQTWithShift
from backbone import OctaveNet


class LKEModel(nn.Module):
    """
    The model for local key estimation

    Parameters
    ----------
    feature_args : dict
        Arguments for the CQT feature layer
    model_args : dict
        Arguments for the OctaveNet backbone
    """

    def __init__(
        self,

        feature_args: dict,
        model_args: dict,

    ) -> None:
        super().__init__()

        self.feature_layer = CQTWithShift(**feature_args)
        self.backbone = OctaveNet(**model_args)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def train_step(self, batch):
        """
        Training step

        Input
        ----------
        batch : dict
            A batch of data containing "x" and "y"

        Output
        ----------
        loss : torch.Tensor
            The loss value
        y_pred : torch.Tensor
            The predicted labels, shape (batch, num_frames, num_classes)
        """
        x, y = batch["x"], batch["y"]
        x, y = self.feature_layer(x, y, shift=True)

        y_pred = self.backbone(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred

    def val_step(self, batch):
        """
        Validation step

        Input and output are the same as train_step
        """
        x, y = batch["x"], batch["y"]
        x, y = self.feature_layer(x, y, shift=False)

        y_pred = self.backbone(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred

    def pred_step(self, batch):
        """
        Prediction step

        Input and output are the same as train_step
        """
        x, y = batch["x"], batch["y"]
        x, y = self.feature_layer(x, y, shift=False)

        return self.backbone(x)

    def forward(self, batch):
        """
        Same as pred_step
        """
        return self.pred_step(batch)
