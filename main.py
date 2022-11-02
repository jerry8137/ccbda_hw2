import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import brainDataset
from resnet_simclr import ResNetSimCLR
from loss import nt_xent
from knn import KNN

import ipdb


class SimCLR(pl.LightningModule):
    def __init__(self, hparams):
        super(SimCLR, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = ResNetSimCLR(hidden_dim=512, out_dim=512)
        self.loss = nt_xent()

    def setup(self, stage):
        self.train_dataset = brainDataset(root=self.hparams.root_dir,
                                          mode='unlabeled')
        self.val_dataset = brainDataset(root=self.hparams.root_dir,
                                        mode='test')

    def forward(self, x):
        out = self.model(x)

        return out

    def configure_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_dataloader),
            eta_min=0,
            last_epoch=-1
        )

        return [self.optimizer], [self.lr_scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True,
                          shuffle=True,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True,
                          shuffle=False,
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True,
                          shuffle=False,
                          num_workers=self.hparams.num_workers)

    def training_step(self, batch, batch_idx):
        log = {'lr': (self.optimizer).param_groups[0]['lr']}
        input = batch
        embedding = self(input)
        log['loss'] = loss = self.loss(embedding, input)
        self.log('train/loss', loss)
        self.log('lr', log['lr'])

    def validation_step(self, batch, batch_idx):
        input, label = batch
        embedding = self(input)
        acc = KNN(embedding, label, batch_size=len(label))
        log = {'_val_acc': acc}
        return log

    def validation_epoch_end(self, outputs):
        mean_acc = torch.cat(
            [x['val_acc'] for x in outputs],
            dim=0
        ).to(torch.float).mean().item()
        self.log('val/accuracy', mean_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass


def main():
    hparams = get_ops()
    system = SimCLR(hparams)
    wandb_logger = WandbLogger(project='SimCLR')
    checkpoint_callback = ModelCheckpoint(
        dirpath='./ckpt',
        filename=hparams.exp_name+'-{epoch}--{val/accuracy:.2f}',
        monitor='val/accuracy',
        mode='max',
        save_top_k=5,
    )
    trainer = pl.Trainer(
        max_epochs=hparams.num_epoch,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        gpus=1,
    )
    trainer.fit(system,
                ckpt_path=hparams.weight)


if __name__=='__main__':
    main()
