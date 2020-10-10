"""main.py"""

import argparse
import numpy as np
from utils import str2bool
import warnings
import os
import visdom
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid
from utils import cuda, grid2gif
from model_share import Generator_fc
from torchvision import transforms
from PIL import Image
import random
import joblib



warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Solver(object):
    def __init__(self, args):
        self.start_img_path = args.start_img_path
        self.num_latents = args.num_latents
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.viz_port = args.viz_port
        self.viz = visdom.Visdom(port=self.viz_port)
        self.target_latents_path = os.path.join(args.target_latents_path, 'k_value_' + str(args.num_latents))
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

        self.transforms = transforms.Compose([
                            transforms.Resize(args.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
                        ])

        self.output_dir = args.output_dir


    def encode_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img).unsqueeze(0)
        img = Variable(cuda(img, self.use_cuda))
        recon, z = self.Autoencoder(img)
        z_1 = z[:, 0:self.z_size_start_dim]  # 0-200
        z_2 = z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
        z_3 = z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
        z_4 = z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
        z_5 = z[:, self.z_style_start_dim:]  # 80-100
        return img, recon, z_1, z_2, z_3, z_4, z_5


    def generate_colors(self):
        start_img, start_recon, start_z_1, start_z_2, start_z_3, start_z_4, start_z_5 = self.encode_image(self.start_img_path)

        images = {}
        images['start_img'] = start_img.data
        images['start_recon'] = start_recon.data

        for i in range(self.num_latents):
            target_latent = os.path.join(self.target_latents_path, 'latent_'+str(i)+'_'+str(self.num_latents)+'.npy')
            z_new = torch.reshape(torch.from_numpy(np.load(target_latent)).cuda(), (1, self.z_back_color_dim))
            boundary_img_z = torch.cat((start_z_1, start_z_2, start_z_3, z_new.float(), start_z_5), dim=1)
            mid_boundary_img = self.Autoencoder.fc_decoder(boundary_img_z)
            mid_boundary_img = mid_boundary_img.view(mid_boundary_img.shape[0], 256, 8, 8)
            boundary_img = self.Autoencoder.decoder(mid_boundary_img)
            images[str(i)] = F.sigmoid(boundary_img).data

        self.viz_images(images)


    def viz_images(self, images):
        image_list = []
        name_list = []
        for name, img in images.items():
            x = make_grid(img[:100], normalize=True)
            image_list.append(x)
            name_list.append(name)

        images = torch.stack(image_list, dim=0).cpu()
        self.viz.images(images, env='groupby_scores', opts=dict(title=str(self.num_latents)), nrow=10)
        self.save_sample_img(images, name_list)


    def save_sample_img(self, tensor, name_list):
        unloader = transforms.ToPILImage()
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it

        path = os.path.join(self.output_dir, 'k_value_' + str(self.num_latents))
        if not os.path.exists(path):
            os.makedirs(path)

        for idx, img in enumerate(image):
            image_ori = img.squeeze(0)  # remove the fake batch dimension
            image_ori = unloader(image_ori)
            image_ori.save(os.path.join(path, '{}.png'.format(name_list[idx])))



def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)
    net.generate_colors()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    parser.add_argument('--start_img_path',
                        default='/lab/tmpig23b/u/yao-data/dataset/change_backcolor/green/C_large_chocolate_green_dejavusansmono.png',
                        type=str, help='the image which we want to start our changing')
    parser.add_argument('--target_latents_path',
                        default='/lab/tmpig23b/u/yao-data/generativeAE/find_colors/groupby_scores/center_class_center',
                        type=str, help='the path to load the target latent codes')
    parser.add_argument('--num_latents', default=15, type=int, help='the number of latents we group by')
    # parser.add_argument('--end_img_path',
    #                     default='/lab/tmpig23b/u/yao-data/dataset/change_backcolor/blue/A_large_purple_blue_liberationmono.png',
    #                     type=str, help='the image which we want to end our changing')
    parser.add_argument('--output_dir', default='/lab/tmpig23b/u/yao-data/generativeAE/change_backcolor/group_find_colors',
                        type=str, help='output directory')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    # model params
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
    '''
    save model
    '''
    parser.add_argument('--model_save_dir', default='/lab/tmpig23b/u/zhix/interpolation/checkpoints/fonts_Nswap',
                        type=str, help='the directory to load model')
    parser.add_argument('--resume_model_iters', type=int, default=970000, help='resume model from this step')


    args = parser.parse_args()
    main(args)