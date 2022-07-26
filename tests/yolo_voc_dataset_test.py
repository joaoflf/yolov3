import config
import pytest
import torch
from dataset import YoloVOCDataset
from torch.utils.data import DataLoader


@pytest.fixture(scope="session")
def dataloader() -> DataLoader:
    train_dataset = YoloVOCDataset(
        csv_file=config.DATASET_PATH + "1examples.csv",
        image_dir=config.IMAGES_PATH,
        label_dir=config.LABELS_PATH,
        transform=config.transform,
    )
    return DataLoader(train_dataset, batch_size=1, num_workers=6)


def test_data_shape(dataloader):
    assert next(iter(dataloader))[1][0].shape == torch.Size([1, 3, 13, 13, 6])
    assert next(iter(dataloader))[1][1].shape == torch.Size([1, 3, 26, 26, 6])
    assert next(iter(dataloader))[1][2].shape == torch.Size([1, 3, 52, 52, 6])
