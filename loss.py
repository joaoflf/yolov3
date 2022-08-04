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

        b_w_h = torch.exp(predictions[..., 3:5]) * anchors  # as in paper
        print(predictions[..., 3:5].shape)
        # get the indices of the boxes with objects to find the cell numbers
        object_boxes = obj.nonzero().squeeze().tolist()
        c_x_y = torch.tensor([box[2:4] for box in object_boxes])
        print(c_x_y.shape)
        c_x_y = c_x_y[None, None, None, :]
        print(self.sigmoid(predictions[..., 1:3]).shape, c_x_y.shape)
        b_x_y = self.sigmoid(predictions[..., 1:3]) + c_x_y
        # get the indices of the boxes with objects to find the cell numbers
        # object_boxes = obj.nonzero().squeeze().tolist()
        # anchors_with_origin = torch.tensor(
        #     [[0.5, 0.5] + list(anchors[object_box[1]]) for object_box in object_boxes]
        # )
        # print(targets[..., 1:5][obj], anchors_with_origin)
        # print(box_iou(targets[..., 1:5][obj], anchors_with_origin))
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
