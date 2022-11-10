import torch
from torch.utils.data import DataLoader

import config
from dataset import YoloVOCDataset
from loss import YoloV3Loss
from model import YoloV3
from trainer import Trainer

# from utils import plot_labels, plot_prediction
from utils import get_mAP, plot_labels, plot_prediction

if __name__ == "__main__":

    dataset = YoloVOCDataset(
        csv_file=config.DATASET_PATH + "2examples.csv",
        image_dir=config.IMAGES_PATH,
        label_dir=config.LABELS_PATH,
        transform=config.test_transform,
    )
    loader = DataLoader(dataset, batch_size=2, num_workers=2)
    loss_fn = YoloV3Loss().to(config.DEVICE)
    model = YoloV3(20).to(config.DEVICE)
    optimizer = torch.optim.Adam
    epochs = 500
    lr = 0.001
    batch_size = 2
    checkpoint_path = config.CHECKPOINT_PATH
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(
            [config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8]
        )
        .unsqueeze(1)
        .unsqueeze(1)
        .repeat(1, 3, 2)
    ).to(config.DEVICE)

    trainer = Trainer(
        model,
        loader,
        loss_fn,
        scaled_anchors,
        optimizer,
        epochs,
        lr,
        batch_size,
        checkpoint_path,
    )
    # trainer.train()
    trainer.load_checkpoint(checkpoint_path)
    images, labels = next(iter(loader))

    model.eval()
    predictions = model(images.to(config.DEVICE))
    #    check_accuracy(predictions, labels)
    # plot_prediction(images, predictions, 1)
    # plot_labels(images, labels, 1)
    mAP = get_mAP(predictions, labels)
    print(mAP)
