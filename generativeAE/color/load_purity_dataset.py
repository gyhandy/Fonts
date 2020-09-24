import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def data_loader(batch_size=1, workers=1, pin_memory=True):
    traindir = "/lab/tmpig23b/u/yao-data/generationAE/dataset/purity"
    normalize = transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                     std=[0.225, 0.225, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.ToTensor(),
            #normalize
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader