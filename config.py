import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "./PASCAL_VOC/"
IMAGES_PATH = DATASET_PATH + "images/"
LABELS_PATH = DATASET_PATH + "labels/"
IMAGE_SIZE = 416
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
CELL_SIZES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
ANCHORS_PER_GRID_SIZE = 3
IGNORE_IOU_THRESHOLD = 0.5

transform = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * 1.2)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * 1.2),
            min_width=int(IMAGE_SIZE * 1.2),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, saturation=0.6, contrast=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.5
                ),
                A.Affine(shear=15),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.ToGray(p=0.1),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=[], min_visibility=0.4),
)

test_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

display_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)
