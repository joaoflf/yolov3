from tkinter import W

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


def nms(
    boxes: torch.Tensor, confidence_threshold: float = 0.7, iou_threshold: float = 0.5
) -> torch.Tensor:
    boxes = boxes[boxes[:, 0] > confidence_threshold]
    sorted_boxes, idx = torch.sort(boxes, 0, descending=True)
    sorted_boxes = sorted_boxes.tolist()
    suppressed_boxes = []

    while len(sorted_boxes) > 0:
        current_box = sorted_boxes.pop(0)
        suppressed_boxes.append(current_box)
        sorted_boxes = [
            box
            for box in sorted_boxes
            if (
                intersection_over_union(
                    torch.tensor(current_box[1:5]), torch.tensor(box[1:5])
                )
                < iou_threshold
            )
        ]
    return torch.tensor(suppressed_boxes)


def cell_to_image_coords(
    cell_size: int, cell_coord: torch.Tensor, x_y: torch.Tensor, w_h: torch.Tensor
) -> torch.Tensor:
    cell_coord = cell_coord * (config.IMAGE_SIZE // cell_size)
    x_y = (x_y * (config.IMAGE_SIZE // cell_size) + cell_coord) / config.IMAGE_SIZE
    w_h = w_h / (config.IMAGE_SIZE // cell_size)
    return torch.cat([x_y, w_h])


def plot_prediction(image, predictions: torch.Tensor, pred_no: int):
    fig, ax = plt.subplots()
    ax.imshow(image[pred_no].permute(1, 2, 0))
    boxes_by_class = {}
    for scale in range(3):

        obj_indices = torch.sigmoid(predictions[scale][pred_no][..., 0]) > 0.8
        obj_preds = predictions[scale][pred_no][..., 0:][obj_indices]
        obj_cell_indices = obj_indices.nonzero()
        for i, obj_pred in enumerate(obj_preds):
            obj_class = torch.argmax(obj_pred[5:]).item()
            x_y = torch.sigmoid(obj_pred[1:3])
            anchors = (
                torch.tensor(config.ANCHORS[scale]).to(config.DEVICE)
                * config.CELL_SIZES[scale]
            )
            w_h = torch.exp(obj_pred[3:5]) * anchors[obj_cell_indices[i][0]]
            obj_coords = cell_to_image_coords(
                config.CELL_SIZES[scale], obj_cell_indices[i][1:3], x_y, w_h
            )
            obj_with_score = [[torch.sigmoid(obj_pred[0]).item(), *obj_coords.tolist()]]
            if obj_class not in boxes_by_class:
                boxes_by_class[obj_class] = obj_with_score
            else:
                boxes_by_class[obj_class] += obj_with_score

    for class_no in boxes_by_class.keys():
        suppressed_boxes = nms(torch.tensor(boxes_by_class[class_no]), 0.8, 0.4)
        for box in suppressed_boxes:
            draw_box(box[0:5], ax, "g", image.shape[2], image.shape[3], "green")
    for box in boxes_by_class[14]:
        if box[0] > 0.9:
            draw_box(
                torch.tensor(box[0:5]), ax, "r", image.shape[2], image.shape[3], "red"
            )

    plt.show()


def plot_labels(image: torch.Tensor, labels: torch.Tensor, label_no: int):
    fig, ax = plt.subplots()
    ax.imshow(image[label_no].permute(1, 2, 0))
    for scale in range(3):
        labels_obj = labels[scale][label_no][..., 0] == 1
        label_data = labels[scale][label_no][..., 0:6][labels_obj]

        obj_cell_indices = labels_obj.nonzero()
        for i, obj_label in enumerate(label_data):
            obj_class = obj_label[5].item()
            obj_coord = cell_to_image_coords(
                config.CELL_SIZES[scale],
                obj_cell_indices[i][1:3],
                obj_label[1:3],
                obj_label[3:5],
            )
            draw_box(obj_coord, ax, "r", image.shape[2], image.shape[3])
    plt.show()


def center_to_edge_coords(boxes: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [
            [
                coords[0] - coords[2] / 2,
                coords[1] - coords[3] / 2,
                coords[0] + coords[2],
                coords[1] + coords[3],
            ]
            for coords in boxes
        ]
    )


def draw_box(
    coords: torch.Tensor,
    axes: axes.Axes,
    color: str,
    image_width: float,
    image_height: float,
    label_color: str,
):
    label, x, y, width, height = coords.tolist()
    rect = patches.Rectangle(
        ((x - width / 2) * image_width, (y - height / 2) * image_height),
        width * image_width,
        height * image_height,
        linewidth=1,
        edgecolor=color,
        facecolor="none",
    )
    axes.add_patch(rect)

    rx, ry = rect.get_xy()
    cx = rx + rect.get_width() / 2.0
    cy = ry + rect.get_height() / 2.0
    axes.annotate(
        round(label, 3),
        (cx, cy),
        color=label_color,
        fontsize=6,
        ha="center",
        va="center",
    )
