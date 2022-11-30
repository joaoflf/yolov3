from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import wandb


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        scaled_anchors: torch.Tensor,
        optimizer: Callable,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        checkpoint_path: str,
    ):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.scaled_anchors = scaled_anchors
        self.optimizer = optimizer(model.parameters(), learning_rate)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.checkpoint_path = checkpoint_path
        self.scaler = torch.cuda.amp.GradScaler()
        wandb.config = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        }

    def train(self):
        wandb.init(project="yolov3")
        looper = tqdm(range(self.epochs))
        loss = 0
        for epoch in looper:
            for index, (image, labels) in enumerate(self.dataloader):
                loss = self.train_step(image, labels)
                looper.set_postfix_str(str(loss))
                wandb.log({"loss": loss})
            self.save_checkpoint(epoch + 1, loss)

    def train_step(self, image: torch.Tensor, labels: torch.Tensor) -> float:
        outputs = self.model(image.to(config.DEVICE))
        loss = (
            self.loss_fn(
                outputs[0].to(config.DEVICE),
                labels[0].to(config.DEVICE),
                self.scaled_anchors[0],
            )
            + self.loss_fn(
                outputs[1].to(config.DEVICE),
                labels[1].to(config.DEVICE),
                self.scaled_anchors[1],
            )
            + self.loss_fn(
                outputs[2].to(config.DEVICE),
                labels[2].to(config.DEVICE),
                self.scaled_anchors[2],
            )
        )

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def save_checkpoint(self, epoch, loss):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            self.checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(
            f"Loaded checkpoint trained on {checkpoint['epoch']} epochs with a loss of {checkpoint['loss']}"
        )
