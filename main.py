import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import YoloVOCDataset
from model import CNNBlock, YoloV3


def main():

    dataset = YoloVOCDataset(
        csv_file=config.DATASET_PATH + "1examples.csv",
        image_dir=config.IMAGES_PATH,
        label_dir=config.LABELS_PATH,
        transform=config.transform,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=6)
    model = YoloV3()
    for index, (image, label) in enumerate(loader):
        model(image)


if __name__ == "__main__":
    main()
