import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_msssim import ms_ssim


def gradient_loss(pred, tgt):
    gx_p = pred[:, :, :, :-1] - pred[:, :, :, 1:]
    gy_p = pred[:, :, :-1, :] - pred[:, :, 1:, :]
    gx_t = tgt[:, :, :, :-1] - tgt[:, :, :, 1:]
    gy_t = tgt[:, :, :-1, :] - tgt[:, :, 1:, :]
    return (gx_p - gx_t).abs().mean() + (gy_p - gy_t).abs().mean()


def charbonnier_loss(pred, tgt, eps=1e-3):
    diff = pred - tgt
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def mix_loss(
    pred,
    tgt,
    alpha=0.84,  # MS-SSIM weight
    beta=0.05,  # Gradient weight
    gamma=0.01,  # Charbonnier weight
    win_size=7,
    data_range=1.0,
):
    # MS-SSIM
    msssim_val = ms_ssim(
        pred, tgt, data_range=data_range, size_average=True, win_size=win_size
    )
    loss_ms = 1 - msssim_val

    # L1
    loss_l1 = nn.L1Loss(pred, tgt, reduction="mean")

    # Gradient
    loss_grad = gradient_loss(pred, tgt)

    # Charbonnier
    loss_char = charbonnier_loss(pred, tgt)

    # Weighted sum
    loss = (
        alpha * loss_ms + (1 - alpha) * loss_l1 + beta * loss_grad + gamma * loss_char
    )
    return loss, msssim_val, loss_l1, loss_grad, loss_char


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        self.alpha = 0.84
        self.beta = 0.05
        self.gamma = 0.01

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss, msssim_val, loss_l1, loss_grad, loss_char = mix_loss(
            restored,
            clean_patch,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            win_size=7,
            data_range=1.0,
        )

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/ms_ssim", msssim_val, prog_bar=True)
        self.log("train/l1", loss_l1, prog_bar=False)
        self.log("train/grad", loss_grad, prog_bar=False)
        self.log("train/charb", loss_char, prog_bar=False)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=30, max_epochs=251
        )

        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir, every_n_epochs=10, save_top_k=-1
    )
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
    )

    model = PromptIRModel()

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == "__main__":
    main()
