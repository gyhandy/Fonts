"""dataset.py"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import transforms as T
from image_folder import make_dataset
from PIL import Image
from PIL import ImageFile
import random

def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)
class ilab_Nswap_imgfolder(Dataset):
    '''
    Content / size / color(Font) / color(background) / style
    E.g. A / 64/ red / blue / arial
    C random sample
    AC same content; BC same size; DC same font_color; EC same back_color; FC same style
    '''
    def __init__(self, root, transform=None):
        super(ilab_Nswap_imgfolder, self).__init__()
        self.root = root
        self.transform = transform
        self.paths = make_dataset(self.root)


    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        # random choose a C image
        # index = random.randint(0, 51)
        # C_letter  = self.Letters[index]
        # C_size = random.choice(self.Sizes)
        # C_font_color = random.choice(self.Colors)
        # resume_colors = self.Colors.copy()
        # resume_colors.remove(C_font_color)
        # C_back_color = random.choice(resume_colors)
        # C_font = random.choice(self.All_fonts)
        # C_img_name = C_letter + '_' + C_size + '_' + C_font_color + '_' + C_back_color + '_' + C_font + ".png"
        # C_img_path = os.path.join(self.root, C_letter, C_size, C_font_color, C_back_color, C_font, C_img_name)

        C_img_path = self.paths[index]
        C_img = Image.open(C_img_path).convert('RGB')



        if self.transform is not None:

            C = self.transform(C_img)

        return C

    def __len__(self):
        # return self.C_size
        # return 500000
        return len(self.paths)


def return_data(args):
    batch_size = 256
    num_workers = 16
    image_size = 64

    root = '/home2/fonts_half_onedir'
    # root = '/home2/fonts_dataset_half'
    if not os.path.exists(root):
        print('No fonts dataset')
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    # dataset = ilab_sup_imgfolder(root, transform)
    # dataset = ilab_threeswap_imgfolder(root, transform)
    dataset = ilab_Nswap_imgfolder(root, transform)



    # train_data = dset(**train_kwargs)
    # train_loader = DataLoader(train_data,
    #                           batch_size=batch_size,
    #                           shuffle=True,
    #                           num_workers=num_workers,
    #                           pin_memory=True,
    #                           drop_last=True)
    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)


    return data_loader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),])

    dset = CustomImageFolder('data/CelebA', transform)
    loader = DataLoader(dset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=False,
                       drop_last=True)

    images1 = iter(loader).next()
    import ipdb; ipdb.set_trace()
