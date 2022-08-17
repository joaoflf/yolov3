import torch
from matplotlib import axes, patches
from matplotlib import pyplot as plt
from PIL import Image

import config


def iou_width_height(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(
    boxes_preds: torch.Tensor, boxes_labels: torch.Tensor
) -> torch.Tensor:
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def cell_to_image_coords(
    cell_size: int, cell_coord: torch.Tensor, box_coord: torch.Tensor
) -> torch.Tensor:
    cell_coord = cell_coord * (config.IMAGE_SIZE / cell_size)
    x_y = (box_coord[0:2] * cell_size + cell_coord) / config.IMAGE_SIZE
    w_h = box_coord[2:4] / cell_size
    return torch.cat([x_y, w_h])


def plot_predictions(image, labels: torch.Tensor, predictions: torch.Tensor):
    labels_obj = labels[0][..., 0] == 1
    labels_object_indexes = labels_obj.nonzero().squeeze()
    labels_boxes = labels[0][..., 1:5][labels_obj]
    pred_obj = labels[0][..., 0] == 1
    pred_object_indexes = pred_obj.nonzero().squeeze()
    pred_boxes = labels[0][..., 1:5][pred_obj]
    fig, ax = plt.subplots()
    ax.imshow(image[0])
    for i, bbox in enumerate(labels_boxes):
        converted_box = cell_to_image_coords(
            config.CELL_SIZES[labels_object_indexes[i][0]],
            labels_object_indexes[i][2:5],
            bbox,
        )
        rect = draw_box(converted_box, ax, "r")

    plt.show()


def draw_box(coords: torch.Tensor, axes: axes.Axes, color: str):
    x, y, width, height = coords
    rect = patches.Rectangle(
        ((x - width / 2) * config.IMAGE_SIZE, (y - height / 2) * config.IMAGE_SIZE),
        width * config.IMAGE_SIZE,
        height * config.IMAGE_SIZE,
        linewidth=1,
        edgecolor=color,
        facecolor="none",
    )
    axes.add_patch(rect)
