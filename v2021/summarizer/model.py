import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from torch import optim
# from torchmetrics import F1
from transformers import ViTModel


class SummaryModel(LightningModule):
    def __init__(self, hidden_dim=768, individual_logs=None):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.scorer = nn.Linear(hidden_dim, 1)
        # self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        # self.train_f1 = F1()
        # self.val_f1 = F1()
        # self.test_f1 = F1()
        self.individual_logs = individual_logs
        self.tta_logs = defaultdict(list)

    def forward(self, x):
        x = self.vit(x).pooler_output
        x = self.scorer(x)
        # x = self.sigmoid(x)
        return x

    def run_batch(self, batch, batch_idx, metric, training=False):
        video_name, image_features, labels = batch
        video_name = video_name[0]
        image_features = image_features.squeeze(0)
        labels = labels.squeeze(0)

        # Score - aggregated labels.
        score = torch.sum(labels, dim=0)
        score = torch.min(
            score,
            torch.ones(
                score.shape[0],
            ).to(score.device),
        )
        out = self(image_features).squeeze(1)
        try:
            loss = self.loss(out.double(), score)
            preds = (torch.sigmoid(out) > 0.7).int()
            metric.update(preds, score.int())
            f1 = metric.compute()
            tp, fp, tn, fn = metric._get_final_stats()
            self.tta_logs[video_name].append((tp.item(), fp.item(), fn.item()))
        except Exception as e:
            print(e)
            loss = 0
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx, self.train_f1, training=True)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log("train_f1", self.train_f1.compute())
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx, self.val_f1)
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        self.log("val_f1", self.val_f1.compute())
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx, self.test_f1)
        self.log("test_loss", loss)
        return loss

    def test_epoch_end(self, outputs):
        f1 = self.test_f1.compute()
        self.log("test_f1", f1)
        tp, fp, tn, fn = self.test_f1._get_final_stats()
        print(f"\nTest f1: {f1}, TP: {tp}, FP: {fp}, TN: {tn}, fn: {fn}")
        self.test_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument")
    args = parser.parse_args()
