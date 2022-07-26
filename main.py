import config
import torch
from dataset import YoloVOCDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CNNBlock


def main():

    dataset = YoloVOCDataset(
        csv_file=config.DATASET_PATH + "1examples.csv",
        image_dir=config.IMAGES_PATH,
        label_dir=config.LABELS_PATH,
        transform=config.transform,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=6)
    model = CNNBlock(in_channels=3, out_channels=32, has_bn=True, kernel_size=3)
    for index, (image, label) in enumerate(loader):
        print(model(image).shape)


if __name__ == "__main__":
    main()
