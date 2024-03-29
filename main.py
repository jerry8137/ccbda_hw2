import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import brainDataset
from resnet_simclr import ResNetSimCLR
from loss import nt_xent
from knn import KNN
from utils import get_opts

import ipdb


class SimCLR(pl.LightningModule):
    def __init__(self, hparams):
        super(SimCLR, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = ResNetSimCLR(out_dim=hparams.hidden_size)
        self.loss = nt_xent

    def setup(self, stage):
        self.train_dataset = brainDataset(root=self.hparams.root_dir,
                                          mode='unlabeled')
        self.val_dataset = brainDataset(root=self.hparams.root_dir,
                                        mode='test')
        self.test_dataset = brainDataset(root=self.hparams.root_dir,
                                         mode='unlabeled',
                                         gen=True)

    def forward(self, x):
        out = self.model(x)

        return out

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_dataloader())*self.hparams.accumulate,
            eta_min=0,
            last_epoch=-1
        )

        return [self.optimizer], [self.scheduler]

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
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True,
                          shuffle=False,
                          num_workers=self.hparams.num_workers)

    def training_step(self, batch, batch_idx):
        log = {'lr': (self.optimizer).param_groups[0]['lr']}
        input, aug_input = batch
        embedding, output = self(input)
        aug_embedding, aug_output = self(aug_input)
        log['loss'] = loss = self.loss(output, aug_output)
        self.log('train/loss', loss)
        self.log('lr', log['lr'])
        return loss

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        input, label = batch
        embedding, _ = self(input)
        acc = KNN(embedding, label, batch_size=512)
        log = {'_val_acc': acc}
        return log

    def validation_epoch_end(self, outputs):
        mean_acc = np.array(
            [x['_val_acc'] for x in outputs]
        ).mean()
        self.log('val/accuracy', mean_acc, prog_bar=True)
        self.log('val_accuracy', mean_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        embedding, _ = self(batch)
        return embedding

    def test_epoch_end(self, outputs):
        embedding = torch.cat(outputs).cpu().detach().numpy()
        np.save('311511035.npy', embedding)


def main():
    hparams = get_opts()
    system = SimCLR(hparams)
    wandb_logger = WandbLogger(project='SimCLR')
    checkpoint_callback = ModelCheckpoint(
        dirpath='./ckpt',
        filename=hparams.exp_name+'/{epoch}-{val_accuracy:.2f}',
        monitor='val/accuracy',
        mode='max',
        save_top_k=5,
    )
    if hparams.test:
        trainer = pl.Trainer(
            gpus=1
        )
        trainer.test(system,
                     ckpt_path=hparams.weight)
    else:
        trainer = pl.Trainer(
            max_epochs=hparams.num_epoch,
            callbacks=[checkpoint_callback],
            logger=wandb_logger,
            gpus=1,
            log_every_n_steps=1,
            accumulate_grad_batches=hparams.accumulate,
        )
        trainer.fit(system,
                    ckpt_path=hparams.weight)


if __name__ == '__main__':
    main()
