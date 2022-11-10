from tkinter import W

import torch
from matplotlib import axes, patches
from matplotlib import pyplot as plt
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
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


def non_max_suppression(bboxes, iou_threshold, threshold):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]), torch.tensor(box[2:])
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def get_bboxes(
    preds: torch.Tensor,
    anchors: torch.Tensor,
    cell_size: int,
    is_preds: bool = True,
) -> list:
    with torch.no_grad():
        all_bboxes = []
        preds[..., 0] = torch.sigmoid(preds[..., 0]) if is_preds else preds[..., 0]
        obj_indices = preds[..., 0] > 0.8
        obj_preds = preds[..., 0:][obj_indices]
        obj_cell_indices = obj_indices.nonzero()
        for i, obj_pred in enumerate(obj_preds):
            if is_preds:
                obj_pred[1:3] = torch.sigmoid(obj_pred[1:3])
                obj_pred[3:5] = (
                    torch.exp(obj_pred[3:5]) * anchors[obj_cell_indices[i][0]]
                )
                obj_score = torch.sigmoid(obj_pred[0]).item()
                obj_class = torch.argmax(obj_pred[5:]).item()
            else:
                obj_score = obj_pred[0].item()
                obj_class = obj_pred[5].item()

            obj_coords = cell_to_image_coords(
                cell_size, obj_cell_indices[i][1:3], obj_pred[1:5]
            )
            obj = [obj_class, obj_score, *obj_coords.tolist()]
            all_bboxes += [obj]
    return all_bboxes


def cell_to_image_coords(
    cell_size: int, cell_coord: torch.Tensor, box_coords: torch.Tensor
) -> torch.Tensor:
    cell_coord = cell_coord * (config.IMAGE_SIZE // cell_size)
    box_coords[0:2] = (
        box_coords[0:2] * (config.IMAGE_SIZE // cell_size) + cell_coord
    ) / config.IMAGE_SIZE
    box_coords[2:5] = box_coords[2:5] / (config.IMAGE_SIZE // cell_size)
    return box_coords


def get_mAP(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    with torch.no_grad():
        all_pred_bboxes = {}
        all_label_bboxes = {}
        mAP_preds = []
        mAP_labels = []
        for scale in range(3):
            for i, pred in enumerate(preds[scale]):
                anchors = (
                    torch.tensor(config.ANCHORS[scale]).to(config.DEVICE)
                    * config.CELL_SIZES[scale]
                )
                pred_bboxes = get_bboxes(pred, anchors, config.CELL_SIZES[scale])
                if all_pred_bboxes.get(i) is None:
                    all_pred_bboxes[i] = pred_bboxes
                    label_bboxes = get_bboxes(
                        labels[2][i], torch.tensor([]), config.CELL_SIZES[2], False
                    )
                    all_label_bboxes[i] = label_bboxes
                else:
                    all_pred_bboxes[i] += pred_bboxes

        for pred_bboxes in all_pred_bboxes.values():
            suppressed_bboxes = non_max_suppression(pred_bboxes, 0.4, 0.5)
            mAP_preds += [
                dict(
                    boxes=torch.tensor(
                        list(map(lambda obj: obj[2:6], suppressed_bboxes))
                    ),
                    scores=torch.tensor(
                        list(map(lambda obj: obj[1], suppressed_bboxes))
                    ),
                    labels=torch.tensor(
                        list(map(lambda obj: obj[0], suppressed_bboxes))
                    ),
                )
            ]
        for label_bboxes in all_label_bboxes.values():
            mAP_labels += [
                dict(
                    boxes=torch.tensor(list(map(lambda obj: obj[2:6], label_bboxes))),
                    labels=torch.tensor(list(map(lambda obj: obj[0], label_bboxes))),
                )
            ]

        mAP = MeanAveragePrecision(box_format="cxcywh")
        mAP.update(mAP_preds, mAP_labels)
        return mAP.compute()


def plot_prediction(image, predictions: torch.Tensor, pred_no: int):
    fig, ax = plt.subplots()
    ax.imshow(image[pred_no].permute(1, 2, 0))
    all_boxes = []
    for scale in range(3):
        anchors = (
            torch.tensor(config.ANCHORS[scale]).to(config.DEVICE)
            * config.CELL_SIZES[scale]
        )
        bboxes = get_bboxes(
            predictions[scale][pred_no], anchors, config.CELL_SIZES[scale]
        )
        all_boxes += bboxes

    suppressed_boxes = non_max_suppression(all_boxes, 0.4, 0.5)
    for box in suppressed_boxes:
        draw_box(box[2:6], box[1], ax, "r", image.shape[2], image.shape[3])
    plt.show()


def plot_labels(image: torch.Tensor, labels: torch.Tensor, label_no: int):
    fig, ax = plt.subplots()
    ax.imshow(image[label_no].permute(1, 2, 0))
    scale = 2

    bboxes = get_bboxes(
        labels[scale][label_no], torch.tensor([]), config.CELL_SIZES[scale], False
    )
    for box in bboxes:
        draw_box(box[2:6], box[0], ax, "purple", image.shape[2], image.shape[3])

    plt.show()

    target = [
        dict(
            boxes=torch.tensor(list(map(lambda obj: obj[2:6], bboxes))),
            labels=torch.tensor(list(map(lambda obj: obj[0], bboxes))),
        )
    ]
    return target


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


def edge_to_center_coords(boxes: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [
            [
                (coords[0] + coords[2]) / 2,
                (coords[1] + coords[3]) / 2,
                (coords[2] - coords[0]) / 2,
                (coords[3] - coords[1]) / 2,
            ]
            for coords in boxes
        ]
    )


def draw_box(
    coords: list,
    label: float,
    axes: axes.Axes,
    color: str,
    image_width: float,
    image_height: float,
):
    x, y, width, height = coords
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
        str(round(label, 3)),
        (cx, cy),
        color=color,
        fontsize=6,
        ha="center",
        va="center",
    )
