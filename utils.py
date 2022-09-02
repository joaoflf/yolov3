from tkinter import W
import torch
from matplotlib import axes, patches
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.ops import nms

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
    cell_size: int, cell_coord: torch.Tensor, x_y: torch.Tensor, w_h: torch.Tensor
) -> torch.Tensor:
    cell_coord = cell_coord * (config.IMAGE_SIZE / cell_size)
    x_y = (x_y * (config.IMAGE_SIZE / cell_size) + cell_coord) / config.IMAGE_SIZE
    w_h = w_h / cell_size
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
            w_hs = torch.exp(obj_pred[3:5]) * anchors
            objs_coords = [
                cell_to_image_coords(
                    config.CELL_SIZES[scale], obj_cell_indices[i][1:3], x_y, w_h
                )
                for w_h in w_hs
            ]
            objs_with_score = [
                [torch.sigmoid(obj_pred[0]).item(), *obj_coords.tolist()]
                for obj_coords in objs_coords
            ]
            if obj_class not in boxes_by_class:
                boxes_by_class[obj_class] = objs_with_score
            else:
                boxes_by_class[obj_class] += objs_with_score

    for class_no in boxes_by_class.keys():
        supressed_boxes = nms(
            center_to_edge_coords(torch.tensor(boxes_by_class[class_no])[..., 1:5]),
            torch.tensor(boxes_by_class[class_no])[..., 0],
            iou_threshold=0.35,
        )
        for box in torch.tensor(boxes_by_class[class_no])[supressed_boxes]:
            draw_box(box[1:5], ax, "r", image.shape[2], image.shape[3])

    plt.show()


def check_accuracy(predictions: torch.Tensor, labels: torch.Tensor):
    for scale in range(3):
        anchors = (
            torch.tensor(config.ANCHORS[scale]).to(config.DEVICE)
            * config.CELL_SIZES[scale]
        )
        labels_obj = labels[scale][0][..., 0] == 1
        label_data = labels[scale][0][..., 0:6][labels_obj]
        pred_boxes = predictions[scale][0][..., 1:5][labels_obj]
        pred_classes = [
            torch.argmax(args) for args in predictions[scale][0][..., 5:][labels_obj]
        ]
        pred_boxes[..., 0:2] = torch.sigmoid(pred_boxes[..., 0:2])
        # pred_boxes[..., 2:4] = torch.exp(pred_boxes[..., 2:4]) * anchors.reshape(
        #     3, 1, 1, 2
        # )
        pred_boxes


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
):
    x, y, width, height = coords.tolist()
    rect = patches.Rectangle(
        ((x - width / 2) * image_width, (y - height / 2) * image_height),
        width * image_width,
        height * image_height,
        linewidth=1,
        edgecolor=color,
        facecolor="none",
    )
    axes.add_patch(rect)
