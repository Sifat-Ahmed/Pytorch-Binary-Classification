import copy
import random
import os
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.models as models
from helper import TqdmUpTo, MetricMonitor, calculate_accuracy
from dataset.utils import get_train_test
from dataset.dataset import SmokeDataset
from visualize import display_image_grid, visualize_augmentations
from config import Config
import warnings
warnings.filterwarnings('ignore')


cudnn.benchmark = True


def train(train_loader, model, criterion, optimizer, epoch, cfg):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(cfg.device, non_blocking=True)
        target = target.to(
            cfg.device, non_blocking=True).float().view(-1, 1)
        output = model(images)
        loss = criterion(output, target)
        accuracy = calculate_accuracy(output, target)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch, metric_monitor=metric_monitor)
        )

def validate(val_loader, model, criterion, epoch, cfg):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(cfg.device, non_blocking=True)
            target = target.to(cfg.device, non_blocking=True).float().view(-1, 1)
            output = model(images)
            loss = criterion(output, target)
            accuracy = calculate_accuracy(output, target)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

def main():
    cfg = Config()
    (train_images, train_labels), (val_images, val_labels), (test_images,
                                                             test_labels) = get_train_test(cfg.dataset_dir)
    #print(len(train_images), len(val_images), len(test_images))

    display_image_grid(train_images, train_labels)

    train_dataset = SmokeDataset(
        train_images, train_labels, resize=cfg.resize, transform=cfg.train_transform)

    val_dataset = SmokeDataset(
        val_images, val_labels, resize=cfg.resize, transform=cfg.val_transform)

    visualize_augmentations(train_dataset, idx=220, name='augments.jpg')

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )


    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )

    model = getattr(models, cfg.model_name)(
        pretrained=False, num_classes=cfg.num_classes,)
    model = model.to(cfg.device)
    criterion = nn.BCEWithLogitsLoss().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    
    for epoch in range(1, cfg.epochs + 1):
        train(train_loader, model, criterion, optimizer, epoch, cfg)
        validate(val_loader, model, criterion, epoch, cfg)
    
    
    torch.save(model.state_dict(), cfg.model_path)
    
    
    
if __name__ == '__main__':
    main()
