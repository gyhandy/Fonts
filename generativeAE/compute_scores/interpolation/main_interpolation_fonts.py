"""main.py"""

import argparse
import numpy as np
from utils import str2bool
import warnings
import os
from tqdm import tqdm
import visdom
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from utils import cuda, grid2gif
from model_share import Generator_fc
from dataset_interpolation_fonts import return_data
from torchvision import transforms
import random



warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.data_loader = return_data(args)
        self.batch_size = args.batch_size
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz = visdom.Visdom(port=self.viz_port)
        self.resume_iters = args.resume_iters
        # model params
        self.image_size = args.image_size
        self.g_conv_dim = args.g_conv_dim
        self.g_repeat_num = args.g_repeat_num
        '''arrangement for each domain'''
        self.z_dim = args.z_dim
        self.z_content_dim = args.z_content
        self.z_size_dim = args.z_size
        self.z_font_color_dim = args.z_font_color
        self.z_back_color_dim = args.z_back_color
        self.z_style_dim = args.z_style

        self.z_content_start_dim = 0
        self.z_size_start_dim = self.z_content_start_dim + self.z_content_dim
        self.z_font_color_start_dim = self.z_size_start_dim + self.z_size_dim
        self.z_back_color_start_dim = self.z_font_color_start_dim + self.z_font_color_dim
        self.z_style_start_dim = self.z_back_color_start_dim + self.z_back_color_dim

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Autoencoder = Generator_fc(3, self.g_conv_dim, self.g_repeat_num, self.z_dim)
        self.Autoencoder.to(self.device)

        Auto_path = os.path.join(args.model_save_dir, '{}-Auto.ckpt'.format(args.resume_model_iters))
        self.Autoencoder.load_state_dict(torch.load(Auto_path, map_location=lambda storage, loc: storage))

        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def encode_image(self, img):
        img = Variable(cuda(img, self.use_cuda))
        recon, z = self.Autoencoder(img)
        z_1 = z[:, 0:self.z_size_start_dim]  # 0-200
        z_2 = z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
        z_3 = z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
        z_4 = z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
        z_5 = z[:, self.z_style_start_dim:]  # 80-100
        return img, recon, z_1, z_2, z_3, z_4, z_5


    def generate_colors(self):
        out = False
        # Start training from scratch or resume training.
        self.global_iter = 0
        if self.resume_iters:
            self.global_iter = self.resume_iters

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)

        pics = ['A', 'B', 'D', 'F']
        while not out:
            for sup_package in self.data_loader:
                images = {}

                labels = sup_package['labels']
                C_img = sup_package['C']
                C_img, C_recon, C_z_1, C_z_2, C_z_3, C_z_4, C_z_5 = self.encode_image(C_img)
                images['C' + '_' + labels['bg']['C'][0]] = C_img.data

                for pic in pics:
                    ori_img = sup_package[pic]
                    ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''
                    ori_img, recon, z_1, z_2, z_3, z_4, z_5 = self.encode_image(ori_img)
                    images[pic + '_' + labels['bg'][pic][0]] = ori_img.data

                    for _ in range(10):
                        z_new = torch.zeros_like(z_4)
                        for m in range(20):
                            for n in range(self.batch_size):
                                z_new[n][m] = random.uniform(z_4[n][m], C_z_4[n][m])
                                interpolation_img_z = torch.cat((z_1, z_2, z_3, z_new, z_5), dim=1)
                                mid_interpolation_img = self.Autoencoder.fc_decoder(interpolation_img_z)
                                mid_interpolation_img = mid_interpolation_img.view(mid_interpolation_img.shape[0], 256, 8, 8)
                                interpolation_img = self.Autoencoder.decoder(mid_interpolation_img)
                                images[labels['bg'][pic][0] + '_' + labels['bg']['C'][0] + str(_)] = F.sigmoid(interpolation_img).data

                self.viz_images(images)
                self.global_iter += 1
                pbar.update(1)


    def viz_images(self, images):
        image_list = []
        name_list = []
        for name, img in images.items():
            x = make_grid(img[:100], normalize=True)
            image_list.append(x)
            name_list.append(name)

        images = torch.stack(image_list, dim=0).cpu()
        self.viz.images(images, env=self.viz_name, opts=dict(title=str(self.global_iter)), nrow=10)
        self.save_sample_img(images, name_list)


    def save_sample_img(self, tensor, name_list):
        unloader = transforms.ToPILImage()
        dir = self.output_dir
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it

        for idx, img in enumerate(image):
            image_ori = img.squeeze(0)  # remove the fake batch dimension
            image_ori = unloader(image_ori)
            image_ori.save(os.path.join(dir, '{}-{}.png'.format(self.global_iter, name_list[idx])))



def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)
    net.generate_colors()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    parser.add_argument('--max_iter', default=5e4, type=float, help='maximum training iteration')
    parser.add_argument('--output_dir', default='/lab/tmpig23b/u/yao-data/generativeAE/dataset/interpolation_center',
                        type=str, help='output directory')
    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')

    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--viz_name', default='interpolation_center', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    # model params
    parser.add_argument('--crop_size', type=int, default=208, help='crop size for the ilab dataset')
    parser.add_argument('--image_size', type=int, default=128, help='crop size for the ilab dataset')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--g_repeat_num', type=int, default=1,
                        help='number of residual blocks in G for encoder and decoder')
    '''
    the weight for pose and background
    '''
    parser.add_argument('--z_dim', default=100, type=int, help='dimension of the representation z')
    parser.add_argument('--z_content', default=20, type=int, help='dimension of the z_content in z')
    parser.add_argument('--z_size', default=20, type=int, help='dimension of the z_size in z')
    parser.add_argument('--z_font_color', default=20, type=int, help='dimension of the z_font_color in z')
    parser.add_argument('--z_back_color', default=20, type=int, help='dimension of the z_back_color in z')
    parser.add_argument('--z_style', default=20, type=int, help='dimension of the z_style in z')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='fonts_unsup_nswap', type=str, help='dataset name')
    parser.add_argument('--num_workers', default=1, type=int, help='dataloader num_workers')

    '''
    save model
    '''
    parser.add_argument('--model_save_dir', default='/lab/tmpig23b/u/zhix/interpolation/checkpoints/fonts_Nswap',
                        type=str, help='the directory to load model')
    parser.add_argument('--resume_model_iters', type=int, default=930000, help='resume model from this step')
    parser.add_argument('--resume_iters', type=int, default=0, help='resume training from this step')
    parser.add_argument('--use_server', default='True', type=str2bool,
                        help='use server to train the model need change the data location')
    parser.add_argument('--which_server', default='15', type=str,
                        help='use which server to train the model 15 or 21')


    args = parser.parse_args()
    main(args)