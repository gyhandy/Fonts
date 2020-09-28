import torchvision.datasets as datasets
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def data_loader(args):

    encode_transforms = transforms.Compose([
                            transforms.Resize(args.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
                        ])

    encode_dataset = datasets.ImageFolder(args.load_images_dir, encode_transforms)
    encode_loader = torch.utils.data.DataLoader(
        encode_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )


    classify_transforms = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                 std=[0.225, 0.225, 0.225])
                        ])

    classify_dataset = datasets.ImageFolder(args.load_images_dir, classify_transforms)
    classify_loader = torch.utils.data.DataLoader(
        classify_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )


    return encode_loader, classify_loader