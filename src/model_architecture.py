from torch import nn
import pytorch_lightning as pl
import torch

class vulcanicModel(pl.LightningModule):
    def __init__(self, features):
        super().__init__()
        self.dense1 = nn.Linear(features, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 32)
        self.dense4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.dense1(x)
        x = nn.functional.relu(x)
        x = self.dense2(x)
        x = nn.functional.relu(x)
        x = self.dense3(x)
        x = nn.functional.relu(x)
        x = self.dense4(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = nn.MSELoss()(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = nn.MSELoss()(y_pred, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

