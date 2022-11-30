import torch
from torch.utils.data import DataLoader

import config
from dataset import YoloVOCDataset
from loss import YoloV3Loss
from model import YoloV3
from trainer import Trainer

# from utils import plot_labels, plot_prediction
from utils import calculate_mAP, plot_labels, plot_prediction

if __name__ == "__main__":

    batch_size = 17
    dataset = YoloVOCDataset(
        csv_file=config.DATASET_PATH + "train.csv",
        image_dir=config.IMAGES_PATH,
        label_dir=config.LABELS_PATH,
        transform=config.transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    loss_fn = YoloV3Loss().to(config.DEVICE)
    model = YoloV3(20).to(config.DEVICE)
    optimizer = torch.optim.Adam
    epochs = 100
    lr = 0.001
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
    trainer.load_checkpoint(checkpoint_path)
    trainer.train()

    # images, labels = next(iter(loader))

    # model.eval()
    # calculate_mAP(model, loader, 0.5, 0.3)
    # predictions = model(images.to(config.DEVICE))
    #    check_accuracy(predictions, labels)
    # plot_prediction(images, predictions, 1)
    # plot_labels(images, labels, 1)
    # mAP = get_mAP(predictions, labels)
    # print(mAP)
