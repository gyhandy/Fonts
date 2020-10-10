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
        self.end_img_path = args.end_img_path
        start_attr = args.start_img_path.split('/')[-2]
        end_attr = args.end_img_path.split('/')[-2]
        self.output_viz_name = start_attr + '_2_' + end_attr
        self.svm = joblib.load(os.path.join(args.boundary_path, args.project_name + end_attr, 'svm_model.m'))


        # start_boundary_path = os.path.join(args.boundary_path, args.project_name + start_attr, 'boundary.npy')
        # end_boundary_path = os.path.join(args.boundary_path, args.project_name + end_attr, 'boundary.npy')
        # self.boundary =  np.load(end_boundary_path) - np.load(start_boundary_path)
        self.boundary = np.zeros([1, args.z_back_color])
        if args.mode == 'target':
            boundaries = []
            for dir in os.listdir(args.boundary_path):
                boundary_path = os.path.join(args.boundary_path, dir, 'boundary.npy')
                if os.path.exists(boundary_path):
                    if dir.split('_')[-1] == start_attr:
                        # self.boundary -= np.load(boundary_path)
                        boundaries.append(np.load(boundary_path))
                    elif dir.split('_')[-1] == end_attr:
                        end_boundary = np.load(boundary_path)
                        self.boundary = end_boundary
                        self.end_mean_attr = np.load(os.path.join(args.boundary_path, dir, 'attribute_mean.npy'))
                    else:
                        boundaries.append(np.load(boundary_path))
            for boundary in boundaries:
                self.boundary -= np.dot(end_boundary, boundary.T) * boundary

            boundary_path = os.path.join(args.boundary_path, args.project_name + end_attr, 'boundary.npy')
            self.end_attr_boundary = np.load(boundary_path)
            self.end_attr_boundary = torch.from_numpy(self.end_attr_boundary).cuda()

        elif args.mode == 'discover':
            for dir in os.listdir(args.boundary_path):
                boundary_path = os.path.join(args.boundary_path, dir, 'boundary.npy')
                if os.path.exists(boundary_path):
                    if dir.split('_')[-1] == start_attr:
                        self.boundary -= np.load(boundary_path)
                    elif dir.split('_')[-1] == end_attr:
                        self.boundary += np.load(boundary_path)
                        self.end_mean_attr = np.load(os.path.join(args.boundary_path, dir, 'attribute_mean.npy'))
                        self.end_attr_boundary = np.load(boundary_path)
                        self.end_attr_boundary = torch.from_numpy(self.end_attr_boundary).cuda()
                    else:
                        self.boundary += np.load(boundary_path)
        self.boundary = self.boundary / np.linalg.norm(self.boundary)


        self.end_mean_attr = torch.from_numpy(self.end_mean_attr).cuda()
        self.boundary = torch.from_numpy(self.boundary).cuda()

        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.viz_port = args.viz_port
        self.viz = visdom.Visdom(port=self.viz_port)
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

        self.output_dir = os.path.join(args.output_dir,self.output_viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


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
        self.global_iter = len(os.listdir(self.output_dir))
        self.iter_output_dir = os.path.join(self.output_dir, str(self.global_iter))


        start_img, start_recon, start_z_1, start_z_2, start_z_3, start_z_4, start_z_5 = self.encode_image(self.start_img_path)
        end_img, end_recon, end_z_1, end_z_2, end_z_3, end_z_4, end_z_5 = self.encode_image(self.end_img_path)

        images = {}
        images['start_img'] = start_img.data
        images['start_recon'] = start_recon.data
        images['end_img'] = end_img.data
        images['end_recon'] = end_recon.data

        ratio = torch.sum(self.end_attr_boundary.mul(self.end_mean_attr - start_z_4), dim=1) / 6
        # ratio = torch.sqrt(torch.sum((self.end_mean_attr - start_z_4).mul(self.end_mean_attr - start_z_4), dim=1)) / 10
        # ratio = torch.max(self.end_mean_attr - start_z_4) / 5
        add_mean_attr_vector = (self.end_mean_attr - start_z_4) / 10
        for i in range(10):
            z_new = start_z_4 + (i + 1) * ratio * self.boundary
            boundary_img_z = torch.cat((start_z_1, start_z_2, start_z_3, z_new.float(), start_z_5), dim=1)
            mid_boundary_img = self.Autoencoder.fc_decoder(boundary_img_z)
            mid_boundary_img = mid_boundary_img.view(mid_boundary_img.shape[0], 256, 8, 8)
            boundary_img = self.Autoencoder.decoder(mid_boundary_img)
            images['combine_boundary' + str(i)] = F.sigmoid(boundary_img).data

            z_new = start_z_4 + (i + 1) * add_mean_attr_vector
            boundary_img_z = torch.cat((start_z_1, start_z_2, start_z_3, z_new.float(), start_z_5), dim=1)
            mid_boundary_img = self.Autoencoder.fc_decoder(boundary_img_z)
            mid_boundary_img = mid_boundary_img.view(mid_boundary_img.shape[0], 256, 8, 8)
            boundary_img = self.Autoencoder.decoder(mid_boundary_img)
            images['mean_attr' + str(i)] = F.sigmoid(boundary_img).data

            z_new = torch.zeros_like(start_z_4)
            for m in range(self.z_back_color_dim):
                z_new[0][m] = random.uniform(start_z_4[0][m], end_z_4[0][m])
            interpolation_img_z = torch.cat((start_z_1, start_z_2, start_z_3, z_new, start_z_5), dim=1)
            mid_interpolation_img = self.Autoencoder.fc_decoder(interpolation_img_z)
            mid_interpolation_img = mid_interpolation_img.view(mid_interpolation_img.shape[0], 256, 8, 8)
            interpolation_img = self.Autoencoder.decoder(mid_interpolation_img)
            images['random_interpolate' + str(i)] = F.sigmoid( interpolation_img).data


        if args.mode == 'target':
            ratio = torch.sum(self.end_attr_boundary.mul(self.end_mean_attr - start_z_4), dim=1) / 5
            for i in range(10):
                z_new = start_z_4 + (i + 1) * ratio * self.end_attr_boundary
                boundary_img_z = torch.cat((start_z_1, start_z_2, start_z_3, z_new.float(), start_z_5), dim=1)
                mid_boundary_img = self.Autoencoder.fc_decoder(boundary_img_z)
                mid_boundary_img = mid_boundary_img.view(mid_boundary_img.shape[0], 256, 8, 8)
                boundary_img = self.Autoencoder.decoder(mid_boundary_img)
                images['single_boundary' + str(i)] = F.sigmoid(boundary_img).data


        if not os.path.exists(self.iter_output_dir):
            os.makedirs(self.iter_output_dir)
        self.viz_images(images)


    def viz_images(self, images):
        image_list = []
        name_list = []
        for name, img in images.items():
            x = make_grid(img[:100], normalize=True)
            image_list.append(x)
            name_list.append(name)

        images = torch.stack(image_list, dim=0).cpu()
        self.viz.images(images, env=self.output_viz_name, opts=dict(title=str(self.global_iter)), nrow=10)
        self.save_sample_img(images, name_list)


    def save_sample_img(self, tensor, name_list):
        unloader = transforms.ToPILImage()
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it

        for idx, img in enumerate(image):
            image_ori = img.squeeze(0)  # remove the fake batch dimension
            image_ori = unloader(image_ori)
            image_ori.save(os.path.join(self.iter_output_dir, '{}.png'.format(name_list[idx])))



def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)
    net.generate_colors()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')
    parser.add_argument('--boundary_path', default='/lab/tmpig23b/u/yao-data/generativeAE/find_colors/boundary_back_color',
                        type=str, help='the directory to load boundary')
    parser.add_argument('--mode', default='discover', type=str,
                        help='change to target color or find more colors (target / discover)')
    parser.add_argument('--project_name', default='center_classRGB_', type=str, help='name to distinguish different boundary')
    parser.add_argument('--start_img_path',
                        default='/lab/tmpig23b/u/yao-data/dataset/change_backcolor/blue/B_large_chocolate_blue_anjalioldlipi.png',
                        type=str, help='the image which we want to start our changing')
    # parser.add_argument('--end_img_path',
    #                     default='/lab/tmpig23b/u/yao-data/dataset/change_backcolor/blue/A_large_purple_blue_rekha.png',
    #                     type=str, help='the image which we want to end our changing')
    # parser.add_argument('--start_img_path',
    #                     default='/lab/tmpig23b/u/yao-data/dataset/change_backcolor/green/a_large_chocolate_green_liberationmono.png',
    #                     type=str, help='the image which we want to start our changing')
    parser.add_argument('--end_img_path',
                        default='/lab/tmpig23b/u/yao-data/dataset/change_backcolor/red/C_large_silver_red_dejavusansmono.png',
                        type=str, help='the image which we want to end our changing')


    parser.add_argument('--output_dir', default='/lab/tmpig23b/u/yao-data/generativeAE/change_backcolor',
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