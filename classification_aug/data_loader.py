import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def data_loader(root, batch_size=8, workers=1, pin_memory=True):
    traindir = os.path.join(root, 'train_small')
    valdir = os.path.join(root, 'test')
    normalize = transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                     std=[0.225, 0.225, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(128),
            #transforms.RandomResizedCrop(128),
            transforms.ColorJitter(0.5,0.5,0.5,0.5),
            transforms.ToTensor(),
            normalize
        ])
    )
    train_dataset2 = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(128),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(128),
            transforms.ToTensor(),
            normalize
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    train_loader2 = torch.utils.data.DataLoader(
        train_dataset2,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, train_loader2
