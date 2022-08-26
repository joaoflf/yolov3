from typing import Callable

import albumentations as A
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import wandb
from dataset import YoloVOCDataset
from loss import YoloV3Loss
from model import YoloV3
from utils import plot_predictions


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Callable,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        checkpoint_path: str,
    ):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
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
        for epoch in looper:
            for index, (image, labels) in enumerate(self.dataloader):
                loss = self.train_step(image, labels)
                looper.set_postfix_str(str(loss))
                wandb.log({"loss": loss})
                if epoch == self.epochs - 1:
                    self.save_checkpoint(epoch + 1, loss)

    def train_step(self, image: torch.Tensor, labels: torch.Tensor) -> float:
        outputs = model(image.to(config.DEVICE))
        loss = (
            self.loss_fn(
                outputs[0].to(config.DEVICE),
                labels[0].to(config.DEVICE),
                torch.tensor(config.ANCHORS[0]).to(config.DEVICE),
            )
            + self.loss_fn(
                outputs[1].to(config.DEVICE),
                labels[1].to(config.DEVICE),
                torch.tensor(config.ANCHORS[1]).to(config.DEVICE),
            )
            + self.loss_fn(
                outputs[2].to(config.DEVICE),
                labels[2].to(config.DEVICE),
                torch.tensor(config.ANCHORS[2]).to(config.DEVICE),
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


if __name__ == "__main__":

    dataset = YoloVOCDataset(
        csv_file=config.DATASET_PATH + "1examples.csv",
        image_dir=config.IMAGES_PATH,
        label_dir=config.LABELS_PATH,
        transform=config.test_transform,
    )
    loader = DataLoader(dataset, batch_size=1)
    loss_fn = YoloV3Loss().to(config.DEVICE)
    model = YoloV3(20).to(config.DEVICE)
    optimizer = torch.optim.Adam
    epochs = 500
    lr = 0.001
    batch_size = 1
    checkpoint_path = config.CHECKPOINT_PATH

    trainer = Trainer(
        model, loader, loss_fn, optimizer, epochs, lr, batch_size, checkpoint_path
    )
    # trainer.train()
    trainer.load_checkpoint(checkpoint_path)
    image, labels = next(iter(loader))

    model.eval()
    predictions = model(image.to(config.DEVICE))
    plot_predictions(image[0], predictions)
