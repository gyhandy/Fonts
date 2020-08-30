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
        # self.paths = make_dataset(self.root)
        self.C_size = 52 # too much we fix it as the number of letters
        '''refer'''
        # color 10
        self.Colors = {'red': (220, 20, 60), 'orange': (255, 165, 0), 'Yellow': (255, 255, 0), 'green': (0, 128, 0),
                  'cyan': (0, 255, 255),
                  'blue': (0, 0, 255), 'purple': (128, 0, 128), 'pink': (255, 192, 203), 'chocolate': (210, 105, 30),
                  'silver': (192, 192, 192)}
        self.Colors = list(self.Colors.keys())
        # size 3
        self.Sizes = {'small': 80, 'medium': 100, 'large': 120}
        self.Sizes = list(self.Sizes.keys())
        # style nearly over 100
        for roots, dirs, files in os.walk(os.path.join(self.root, 'A', 'medium', 'red', 'orange')):
            cates = dirs
            break
        self.All_fonts = cates
        print(len(self.All_fonts))
        print(self.All_fonts, len(self.All_fonts))
        # letter 52
        self.Letters = [chr(x) for x in list(range(65, 91)) + list(range(97, 123))]

    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        # random choose a C image
        C_letter  = self.Letters[index]
        C_size = random.choice(self.Sizes)
        C_font_color = random.choice(self.Colors)
        resume_colors = self.Colors.copy()
        resume_colors.remove(C_font_color)
        C_back_color = random.choice(resume_colors)
        C_font = random.choice(self.All_fonts)
        C_img_name = C_letter + '_' + C_size + '_' + C_font_color + '_' + C_back_color + '_' + C_font + ".png"
        C_img_path = os.path.join(self.root, C_letter, C_size, C_font_color, C_back_color, C_font, C_img_name)


        C_img = Image.open(C_img_path).convert('RGB')



        if self.transform is not None:

            C = self.transform(C_img)

        return C, C_img_name

    def __len__(self):
        return self.C_size

def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    if name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset
    elif name.lower() == 'fonts':
        if args.use_server == True:
            if args.which_server == '15':
                root = '/home2/andy/fonts_dataset_center'
                # root = '/home2/andy/fonts_dataset_half'
            elif args.which_server == '21':
                root = '/home2/andy/fonts_dataset_center'
                # root = '/home2/andy/fonts_dataset_half'
            elif args.which_server == '9':
                # root = '/media/pohsuanh/Data/andy/fonts_dataset_new'
                root = '/media/pohsuanh/Data/andy/fonts_dataset_center'
                # root = '/media/pohsuanh/Data/andy/fonts_dataset_half'
        else:
            root = '/home2/fonts_dataset_center'
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

    else:
        raise NotImplementedError


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
