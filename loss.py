from tkinter import W

import albumentations as A
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

import config
from dataset import YoloVOCDataset
from model import YoloV3
from utils import intersection_over_union as iou
from utils import plot_predictions


class YoloV3Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.no_obj_weight = 0.5
        self.coord_weight = 5

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        obj = targets[..., 0] == 1
        no_obj = targets[..., 0] == 0

        no_obj_loss = self.bce(predictions[..., 0:1][no_obj], targets[..., 0:1][no_obj])

        anchors = anchors[None, :, None, None, :]  # reshape anchors perform operationsj
        b_w_h = torch.exp(predictions[..., 3:5]) * anchors  # as in paper
        b_x_y = self.sigmoid(predictions[..., 1:3][obj])  # + object_cells
        iou_scores = iou(
            torch.cat([b_x_y, b_w_h[obj]], dim=1), targets[..., 1:5][obj]
        ).detach()
        obj_loss = self.mse(iou_scores, self.sigmoid(predictions[..., 0:1][obj]))

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        targets[..., 3:5] = torch.log(1e-16 + targets[..., 3:5] / anchors)
        coord_loss = self.mse(predictions[..., 1:5][obj], targets[..., 1:5][obj])

        return (
            self.no_obj_weight * no_obj_loss + self.coord_weight * coord_loss + obj_loss
        )


if __name__ == "__main__":

    dataset = YoloVOCDataset(
        csv_file=config.DATASET_PATH + "1examples.csv",
        image_dir=config.IMAGES_PATH,
        label_dir=config.LABELS_PATH,
        transform=config.test_transform,
    )
    loader = DataLoader(dataset, batch_size=1)
    # scaler = torch.cuda.amp.GradScaler()
    # loss_fn = YoloV3Loss().to(config.DEVICE)
    # model = YoloV3(20).to(config.DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # epochs = 100
    # for epoch in range(epochs):
    #     for index, (image, labels) in enumerate(loader):
    #         outputs = model(image.to(config.DEVICE))
    #         loss = loss_fn(outputs[0], labels[0], torch.tensor(config.ANCHORS[0]))
    #         loss = (
    #             loss_fn(
    #                 outputs[0].to(config.DEVICE),
    #                 labels[0].to(config.DEVICE),
    #                 torch.tensor(config.ANCHORS[0]).to(config.DEVICE),
    #             )
    #             + loss_fn(
    #                 outputs[1].to(config.DEVICE),
    #                 labels[1].to(config.DEVICE),
    #                 torch.tensor(config.ANCHORS[1]).to(config.DEVICE),
    #             )
    #             + loss_fn(
    #                 outputs[2].to(config.DEVICE),
    #                 labels[2].to(config.DEVICE),
    #                 torch.tensor(config.ANCHORS[2]).to(config.DEVICE),
    #             )
    #         )

    #         optimizer.zero_grad()
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #         print(loss.item())

    #         if epoch == epochs - 1:
    #             pred_boxes = outputs[0][..., 0] == 1
    #             target_boxes = labels[0][..., 0] == 1
    #             print(
    #                 f"prediction: {torch.sigmoid(outputs[0][...,0][target_boxes])} | target: {labels[0][...,0][target_boxes]}"
    #             )

    dataset = YoloVOCDataset(
        csv_file=config.DATASET_PATH + "1examples.csv",
        image_dir=config.IMAGES_PATH,
        label_dir=config.LABELS_PATH,
        transform=config.display_transform,
    )
    loader = DataLoader(dataset, batch_size=1)
    image, labels = next(iter(loader))
    plot_predictions(image, labels)
