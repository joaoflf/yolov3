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
    cell_size: int, cell_coord: torch.Tensor, box_coord: torch.Tensor
) -> torch.Tensor:
    cell_coord = cell_coord * (config.IMAGE_SIZE / cell_size)
    x_y = (box_coord[0:2] * cell_size + cell_coord) / config.IMAGE_SIZE
    w_h = box_coord[2:4] / cell_size
    return torch.cat([x_y, w_h])


def plot_predictions(image, predictions: torch.Tensor):
    fig, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0))
    boxes_by_class = {}
    for scale in range(3):

        labels_obj = predictions[scale][..., 0] > 0.6
        obj_preds = predictions[scale][..., 0:][labels_obj]
        obj_preds_indices = labels_obj.nonzero()
        for i, obj_pred in enumerate(obj_preds):
            obj_class = torch.argmax(obj_pred[5:]).item()
            obj_coords = cell_to_image_coords(
                config.CELL_SIZES[scale], obj_preds_indices[i][2:4], obj_pred[1:5]
            )
            obj_with_score = [obj_pred[0].item(), *obj_coords.tolist()]
            if obj_class not in boxes_by_class:
                boxes_by_class[obj_class] = [obj_with_score]
            else:
                boxes_by_class[obj_class] += [obj_with_score]

    supressed_boxes = nms(
        center_to_edge_coords(torch.tensor(boxes_by_class[12])[..., 1:5]),
        torch.tensor(boxes_by_class[12])[..., 0],
        iou_threshold=0.2,
    )
    for box in torch.tensor(boxes_by_class[12])[supressed_boxes]:
        draw_box(box[1:5], ax, "r", image.shape[1], image.shape[2])

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


def draw_box1(
    coords: torch.Tensor,
    axes: axes.Axes,
    color: str,
    image_width: float,
    image_height: float,
):
    x, y, width, height = coords.tolist()
    rect = patches.Rectangle(
        (x * image_width, y * image_height),
        width - x * image_width,
        height - y * image_height,
        linewidth=1,
        edgecolor=color,
        facecolor="none",
    )
    axes.add_patch(rect)


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
