"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config
from utils import iou_width_height


class YoloVOCDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        label_dir: str,
        transform: Callable,
    ) -> None:
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.anchors = config.ANCHORS
        self.cell_sizes = config.CELL_SIZES
        self.transform = transform
        self.image_size = config.IMAGE_SIZE
        self.ignore_iou_threshold = 0.5

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index) -> tuple:
        image = np.array(
            Image.open(f"{self.image_dir}{self.annotations.iloc[index, 0]}").convert(
                "RGB"
            )
        )
        labels = np.roll(
            np.loadtxt(
                fname=f"{self.label_dir}{self.annotations.iloc[index, 1]}",
                delimiter=" ",
                ndmin=2,
            ),
            4,
            axis=1,
        ).tolist()
        targets = [
            torch.zeros(len(self.cell_sizes), grid_size, grid_size, 6)
            for grid_size in self.cell_sizes
        ]

        augmentations = self.transform(image=image, bboxes=labels)
        image = augmentations["image"]
        labels = augmentations["bboxes"]

        anchors = torch.tensor(self.anchors).flatten(start_dim=0, end_dim=1)

        for bbox in labels:
            x, y, width, height, class_label = bbox

            # calculate height and width iou of bbox with every anchor
            # to determine which anchor shape is more similar to bbox
            iou_bbox_anchors = iou_width_height(torch.tensor(bbox[2:4]), anchors)
            # get sorted anchor indices by iou soore
            sorted_anchor_indices = iou_bbox_anchors.argsort(descending=True)
            bbox_has_anchor = [False] * 3
            for anchor_index in sorted_anchor_indices:
                grid_index = anchor_index.div(3, rounding_mode="floor")
                anchor_in_grid = anchor_index % 3
                grid_x, grid_y = int(x * self.cell_sizes[grid_index]), int(
                    y * self.cell_sizes[grid_index]
                )
                anchor_is_taken = targets[grid_index][anchor_in_grid, grid_x, grid_y, 0]

                if not anchor_is_taken and not bbox_has_anchor[grid_index]:
                    # set objectedness
                    targets[grid_index][anchor_in_grid, grid_x, grid_y, 0] = 1
                    # coordinates relative to cell
                    x_to_cell, y_to_cell = (
                        x * self.cell_sizes[grid_index] - grid_x,
                        y * self.cell_sizes[grid_index] - grid_y,
                    )
                    width_to_cell, height_to_cell = (
                        config.IMAGE_SIZE // self.cell_sizes[grid_index] * width,
                        config.IMAGE_SIZE // self.cell_sizes[grid_index] * height,
                    )

                    targets[grid_index][
                        anchor_in_grid, grid_x, grid_y, 1:5
                    ] = torch.tensor(
                        [x_to_cell, y_to_cell, width_to_cell, height_to_cell]
                    )
                    targets[grid_index][anchor_in_grid, grid_x, grid_y, 5] = int(
                        class_label
                    )
                    bbox_has_anchor[grid_index] = True

                elif (
                    not anchor_is_taken
                    and iou_bbox_anchors[anchor_index] > self.ignore_iou_threshold
                ):
                    targets[grid_index][anchor_in_grid, grid_x, grid_y, 0] = -1

        return image, targets
