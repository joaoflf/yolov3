from tkinter import W

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

import config
from dataset import YoloVOCDataset
from model import YoloV3


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
        b_w_h = torch.exp(targets[..., 3:5]) * anchors  # as in paper

        # get the indices of the boxes with objects to find the cell numbers
        object_cells = obj.nonzero().squeeze()[..., 2:4]
        b_x_y = self.sigmoid(targets[..., 1:3][obj]) + object_cells
        iou = box_iou(torch.cat([b_x_y, b_w_h[obj]], dim=1), targets[..., 1:5][obj])
        obj_loss = self.bce(torch.diag(iou), predictions[..., 0:1][obj].squeeze())
        return predictions


if __name__ == "__main__":
    dataset = YoloVOCDataset(
        csv_file=config.DATASET_PATH + "1examples.csv",
        image_dir=config.IMAGES_PATH,
        label_dir=config.LABELS_PATH,
        transform=config.transform,
    )
    loader = DataLoader(dataset, batch_size=1)
    loss = YoloV3Loss()
    model = YoloV3(20)
    for index, (image, labels) in enumerate(loader):
        outputs = model(image)
        loss(outputs[1], labels[1], torch.tensor(config.ANCHORS[0]))
