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
import fnmatch

ImageFile.LOAD_TRUNCATED_IMAGES = True

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

class ilab_sup_imgfolder(Dataset):
    def __init__(self, root, transform=None):
        super(ilab_sup_imgfolder, self).__init__()
        self.root = root
        self.transform = transform
        self.dir_Appe = os.path.join(self.root, 'appearance')
        self.Appe_paths = make_dataset(self.dir_Appe)
        self.Appe_size = len(self.Appe_paths)

    def __getitem__(self, index):
        Appe_img_path = self.Appe_paths[index % self.Appe_size]
        Pose_img_path = Appe_img_path.replace('appearance', 'pose').replace('appe', 'pose')
        Comb_img_path = Appe_img_path.replace('appearance', 'combine').replace('appe', 'comb')
        Appe_img = Image.open(Appe_img_path).convert('RGB')
        Pose_img = Image.open(Pose_img_path).convert('RGB')
        Comb_img = Image.open(Comb_img_path).convert('RGB')

        if self.transform is not None:
            Appe = self.transform(Appe_img)
            Pose = self.transform(Pose_img)
            Comb = self.transform(Comb_img)

        return {'Appe': Appe, 'Pose': Pose, 'Comb': Comb}

    def __len__(self):
        return self.Appe_size
class ilab_unsup_imgfolder(Dataset):
    def __init__(self, root, transform=None):
        super(ilab_unsup_imgfolder, self).__init__()
        self.root = root
        self.transform = transform
        self.paths = make_dataset(self.root)
        self.C_size = len(self.paths)

    def __getitem__(self, index):
        C_img_path = self.paths[index % self.C_size]
        # C_pose = C_img_path.split('/')[4]
        # C_category = C_img_path.split('/')[5].split('-')[0]
        # C_identity = C_img_path.split('/')[5].split('-')[1]
        C_pose = C_img_path.split('/')[5]
        C_category = C_img_path.split('/')[6].split('-')[0]
        C_identity = C_img_path.split('/')[6].split('-')[1]
        B_root = self.root.replace('vae_unbalance_train', 'vae_unbalance_identity')
        # B has same identity
        B_category = C_category
        B_identity = C_identity
        B_img_root = os.path.join(B_root, B_category, B_identity)
        B_files = os.listdir(B_img_root)
        B_image_index = random.randint(0, len(B_files) - 1)
        B_img_path = os.path.join(B_img_root, B_files[B_image_index])
        # A has same pose
        A_pose = C_pose
        A_img_root = os.path.join(self.root, A_pose)
        A_files = os.listdir(A_img_root)
        A_image_index = random.randint(0, len(A_files) - 1)
        A_img_path = os.path.join(A_img_root, A_files[A_image_index])

        A_img = Image.open(A_img_path).convert('RGB')
        B_img = Image.open(B_img_path).convert('RGB')
        C_img = Image.open(C_img_path).convert('RGB')

        if self.transform is not None:
            A = self.transform(A_img)
            B = self.transform(B_img)
            C = self.transform(C_img)

        return {'A': A, 'B': B, 'C': C}

    def __len__(self):
        return self.C_size
class ilab_threeswap_imgfolder(Dataset):
    def __init__(self, root, transform=None):
        super(ilab_threeswap_imgfolder, self).__init__()
        self.root = root
        self.transform = transform
        self.paths = make_dataset(self.root)
        self.C_size = len(self.paths)

    def findABD(self, index):
        FOUNDED = False
        while not FOUNDED:
            FOUNDED = True
            C_img_path = self.paths[index % self.C_size]
            '''
            local
            '''
            # C_pose_root = C_img_path.split('/')[4]
            # C_pose = C_pose_root.replace('_', '-')
            # C_category = C_img_path.split('/')[5].split('-')[0]
            # C_identity = C_img_path.split('/')[5].split('-')[1]
            # C_back = C_img_path.split('/')[5].split('-')[2]

            '''
            server
            '''
            C_pose_root = C_img_path.split('/')[5]
            C_pose = C_pose_root.replace('_', '-')
            C_category = C_img_path.split('/')[6].split('-')[0]
            C_identity = C_img_path.split('/')[6].split('-')[1]
            C_back = C_img_path.split('/')[6].split('-')[2]

            '''
            server on igpu9
            '''
            # C_pose_root = C_img_path.split('/')[7]
            # C_pose = C_pose_root.replace('_', '-')
            # C_category = C_img_path.split('/')[8].split('-')[0]
            # C_identity = C_img_path.split('/')[8].split('-')[1]
            # C_back = C_img_path.split('/')[8].split('-')[2]

            B_root = self.root.replace('train_img_c00_10class', 'vae_identity_new')
            # B has same identity
            B_category = C_category
            B_identity = C_identity
            B_img_root = os.path.join(B_root, B_category, B_identity)
            # B must have different pose and diff back with C
            B_files = os.listdir(B_img_root)
            B_img_name = random.choice(B_files)
            if not C_pose in B_img_name and not C_back in B_img_name:
                B_img_path = os.path.join(B_img_root, B_img_name)
            else:
                # print('The B image can not have different pose and back with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 1000
                continue  # break the BREAK_ALL

            # A has same pose
            A_img_root = os.path.join(self.root, C_pose_root)
            # A must have different identity and diff back with C
            A_files = os.listdir(A_img_root)
            A_img_name = random.choice(A_files)
            if not C_identity in A_img_name and not C_back in A_img_name:
                A_img_path = os.path.join(A_img_root, A_img_name)
            else:
                # print('The A image can not have different pose and back with C because the C path is {0}'.format(
                #     C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 100
                continue  # break the BREAK_ALL

            # D has same back
            # back-cate-pose
            D_root = self.root.replace('train_img_c00_10class', 'vae_back_new')
            D_back = C_back
            # D has same back
            D_img_root_back = os.path.join(D_root, D_back)
            # D must have different identity and diff pose with C
            '''cate '''
            for roots, dirs, files in os.walk(D_img_root_back):
                cates = dirs
                break
            cates.remove(C_category)
            if len(cates) <= 0:  # no other category to choose
                # print('The D image can not have different cate with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 1 if index < self.C_size - 2 else index - 100
                continue  # break the BREAK_ALL
            selected_D_cate = random.choice(cates)
            D_img_root_cate = os.path.join(D_img_root_back, selected_D_cate)
            '''pose '''
            for roots, dirs, files in os.walk(D_img_root_cate):
                poses = dirs
                break
            # try:
            #     poses.remove(C_pose_root)
            # except:
            #     print(poses, C_pose_root)
            poses.remove(C_pose_root)



            if len(poses) <= 0:  # no other category to choose
                # print('The D image can not have different pose with C because the C path is {0}'.format(C_img_path))
                FOUNDED = False
                index = index + 2 if index < self.C_size - 20 else  index - 200
                continue  # break the BREAK_ALL
            selected_D_pose = random.choice(poses)
            D_img_root = os.path.join(D_img_root_cate, selected_D_pose)
            D_files = os.listdir(D_img_root)
            D_image_index = random.randint(0, len(D_files) - 1)
            D_img_path = os.path.join(D_img_root, D_files[D_image_index])

        return A_img_path, B_img_path, C_img_path, D_img_path
    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        A_img_path, B_img_path, C_img_path, D_img_path = self.findABD(index)


        A_img = Image.open(A_img_path).convert('RGB')
        B_img = Image.open(B_img_path).convert('RGB')
        C_img = Image.open(C_img_path).convert('RGB')
        D_img = Image.open(D_img_path).convert('RGB')


        if self.transform is not None:
            A = self.transform(A_img)
            B = self.transform(B_img)
            C = self.transform(C_img)
            D = self.transform(D_img)

        return {'A': A, 'B': B, 'C': C, 'D': D}

    def __len__(self):
        return self.C_size
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
        self.letter_dict = {}
        self.size_dict = {}
        self.fg_dict = {}
        self.bg_dict1 = {}
        self.style_dict = {}
        self.letter_cnt = 0
        self.size_cnt = 0
        self.fg_cnt = 0
        self.bg_cnt1 = 0
        self.style_cnt = 0
        for roots, dirs, files in os.walk('/home2/andy/fonts_dataset_center'):
            for file in files:
                letter = file.split('_')[0]
                size = file.split('_')[1]
                fg = file.split('_')[2]
                bg = file.split('_')[3]
                style = file.split('_')[4].split('.')[0]
                # print(file)
                if letter not in self.letter_dict:
                    self.letter_dict[letter] = self.letter_cnt
                    self.letter_cnt += 1
                if size not in self.size_dict:
                    self.size_dict[size] = self.size_cnt
                    self.size_cnt += 1
                if fg not in self.fg_dict:
                    self.fg_dict[fg] = self.fg_cnt
                    self.fg_cnt += 1
                if bg not in self.bg_dict1:
                    self.bg_dict1[bg] = self.bg_cnt1
                    self.bg_cnt1 += 1
                if style not in self.style_dict:
                    self.style_dict[style] = self.style_cnt
                    self.style_cnt += 1

    def findN(self, index):
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
        ''' exclusive the C attribute avoid same with C'''
        temp_Letters = self.Letters.copy()# avoid same size with C
        temp_Letters.remove(C_letter)
        temp_Size = self.Sizes.copy()# avoid same size with C
        temp_Size.remove(C_size)
        temp_font_color = self.Colors.copy()# avoid same font_color with C
        temp_font_color.remove(C_font_color)
        temp_back_colors = self.Colors.copy()  # avoid same back_color with C and avoid same color with font
        temp_back_colors.remove(C_back_color)
        temp_font = self.All_fonts.copy()  # avoid same font with C
        temp_font.remove(C_font)

        # A has same content
        '''SAME content'''
        A_letter = C_letter
        A_size = random.choice(temp_Size)
        A_font_color = random.choice(temp_font_color)
        resume_colors = temp_back_colors.copy()
        if A_font_color in resume_colors:
            resume_colors.remove(A_font_color)
        A_back_color = random.choice(resume_colors)
        A_font = random.choice(temp_font)
        A_img_name = A_letter + '_' + A_size + '_' + A_font_color + '_' + A_back_color + '_' + A_font + ".png"
        A_img_path = os.path.join(self.root, A_letter, A_size, A_font_color, A_back_color, A_font, A_img_name)

        # B has same size
        B_letter = random.choice(temp_Letters)
        '''SAME size'''
        B_size = C_size
        B_font_color = random.choice(temp_font_color)
        resume_colors = temp_back_colors.copy()
        if B_font_color in resume_colors:
            resume_colors.remove(B_font_color)
        B_back_color = random.choice(resume_colors)
        B_font = random.choice(temp_font)
        B_img_name = B_letter + '_' + B_size + '_' + B_font_color + '_' + B_back_color + '_' + B_font + ".png"
        B_img_path = os.path.join(self.root, B_letter, B_size, B_font_color, B_back_color, B_font, B_img_name)

        # D has same font_color
        D_letter = random.choice(temp_Letters)
        D_size = random.choice(temp_Size)
        '''SAME font_color'''
        D_font_color = C_font_color
        resume_colors = temp_back_colors.copy()
        if D_font_color in resume_colors:
            resume_colors.remove(D_font_color)
        D_back_color = random.choice(resume_colors)
        D_font = random.choice(temp_font)
        D_img_name = D_letter + '_' + D_size + '_' + D_font_color + '_' + D_back_color + '_' + D_font + ".png"
        D_img_path = os.path.join(self.root, D_letter, D_size, D_font_color, D_back_color, D_font, D_img_name)

        # E has same back_color
        E_letter = random.choice(temp_Letters)
        E_size = random.choice(temp_Size)
        resume_colors = temp_font_color.copy()
        resume_colors.remove(C_back_color)
        E_font_color = random.choice(resume_colors)
        '''SAME back_color'''
        E_back_color = C_back_color
        E_font = random.choice(temp_font)
        E_img_name = E_letter + '_' + E_size + '_' + E_font_color + '_' + E_back_color + '_' + E_font + ".png"
        E_img_path = os.path.join(self.root, E_letter, E_size, E_font_color, E_back_color, E_font, E_img_name)

        # F has same font
        F_letter = random.choice(temp_Letters)
        F_size = random.choice(temp_Size)
        F_font_color = random.choice(temp_font_color)
        resume_colors = temp_back_colors.copy()
        if F_font_color in resume_colors:
            resume_colors.remove(F_font_color)
        F_back_color = random.choice(resume_colors)
        '''SAME font'''
        F_font = C_font
        F_img_name = F_letter + '_' + F_size + '_' + F_font_color + '_' + F_back_color + '_' + F_font + ".png"
        F_img_path = os.path.join(self.root, F_letter, F_size, F_font_color, F_back_color, F_font, F_img_name)
        name_list = [A_img_name, B_img_name, C_img_name, D_img_name, E_img_name, F_img_name]
        return A_img_path, B_img_path, C_img_path, D_img_path, E_img_path, F_img_path, name_list
    def __getitem__(self, index):
        '''there is a big while loop for choose category and training'''
        A_img_path, B_img_path, C_img_path, D_img_path, E_img_path, F_img_path, name_list = self.findN(index)


        A_img = Image.open(A_img_path).convert('RGB')
        B_img = Image.open(B_img_path).convert('RGB')
        C_img = Image.open(C_img_path).convert('RGB')
        D_img = Image.open(D_img_path).convert('RGB')
        E_img = Image.open(E_img_path).convert('RGB')
        F_img = Image.open(F_img_path).convert('RGB')

        labels = {'letter':[], 'size':[], 'bg':[], 'fg':[], 'style':[]}
        for name in name_list:
            labels['letter'].append(self.letter_dict[name.split('_')[0]])
            labels['size'].append(self.size_dict[name.split('_')[1]])
            labels['bg'].append(self.bg_dict1[name.split('_')[3]])
            labels['fg'].append(self.fg_dict[name.split('_')[2]])
            labels['style'].append(self.style_dict[name.split('_')[4].split('.')[0]])


        if self.transform is not None:
            A = self.transform(A_img)
            B = self.transform(B_img)
            C = self.transform(C_img)
            D = self.transform(D_img)
            E = self.transform(E_img)
            F = self.transform(F_img)

        return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'labels':labels}

    def __len__(self):
        return self.C_size

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    train = args.train
    crop_size = args.crop_size
    image_size = args.image_size
    # assert image_sizeimage_size == 64, 'currently only image size of 64 is supported'

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

    elif name.lower() == 'ilab':
        root = '/lab/andy/PycharmProjects/PurePoseGan/stargan/data/train_img_c00_10class'
        if not os.path.exists(root):
            print('No ilab dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        dataset = ImageFolder(root, transform)
    elif name.lower() == 'ilab_sup' or name.lower() == 'ilab_unsup':
        if args.use_server == True:
            root = '/home2/andy/ilab2M_pose/vae_unsup'
        else:
            root = '/home2/iLab-2M-Light/sampled/'
        if not os.path.exists(root):
            print('No ilab_sup dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = ilab_unsup_imgfolder(root, transform)
    elif name.lower() == 'ilab_unsup_unbalance':
        if args.use_server == True:
            root = '/home2/andy/ilab2M_pose/vae_unbalance_tri'
        else:
            root = '/home2/ilab2M_pose/vae_unbalance_tri'
        if not os.path.exists(root):
            print('No ilab_sup dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = ilab_unsup_imgfolder(root, transform)
    elif name.lower() == 'ilab_unsup_unbalance_free':
        if args.use_server == True:
            root = '/home2/andy/ilab2M_pose/vae_unbalance_train'
        else:
            root = '/home2/ilab2M_pose/vae_unbalance_train'
        if not os.path.exists(root):
            print('No ilab_sup dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = ilab_unsup_imgfolder(root, transform)
    elif name.lower() == 'ilab_unsup_unbalance_free_largegap':
        if args.use_server == True:
            root = '/media/home2/andy/dataset/vae_unbalance_train_largegap'
        else:
            root = '/home2/ilab2M_pose/vae_unbalance_train'
        if not os.path.exists(root):
            print('No ilab_sup dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = ilab_unsup_imgfolder(root, transform)
    elif name.lower() == 'ilab_unsup_threeswap':
        if args.use_server == True:
            if args.which_server == '15':
                root = '/home2/andy/ilab2M_pose/train_img_c00_10class'
            elif args.which_server == '21':
                root = '/home2/andy/ilab2M_pose/train_img_c00_10class'
            elif args.which_server == '9':
                root = '/media/pohsuanh/Data/andy/ilab2M_pose/train_img_c00_10class'
        else:
            root = '/home2/ilab2M_pose/train_img_c00_10class'
        if not os.path.exists(root):
            print('No ilab_sup dataset')
        transform = []
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        dataset = ilab_threeswap_imgfolder(root, transform)
    elif name.lower() == 'fonts_unsup_nswap':
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
        if train:
            # transform.append(T.RandomHorizontalFlip()) # Pose information are too sensitive for the flip
            # transform.append(T.CenterCrop(crop_size))
            transform.append(T.Resize(image_size))
            transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
        # dataset = ilab_sup_imgfolder(root, transform)
        # dataset = ilab_threeswap_imgfolder(root, transform)
        dataset = ilab_Nswap_imgfolder(root, transform)



    else:
        raise NotImplementedError


    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=train,
                                  num_workers=num_workers)

    # train_data = dset(**train_kwargs)
    # train_loader = DataLoader(train_data,
    #                           batch_size=batch_size,
    #                           shuffle=True,
    #                           num_workers=num_workers,
    #                           pin_memory=True,
    #                           drop_last=True)

    # data_loader = train_loader

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
