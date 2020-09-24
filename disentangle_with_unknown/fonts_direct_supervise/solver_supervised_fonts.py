"""solver.py"""

import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import visdom

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif
from model_supervised_fonts import Generator_fc, Classifier_Latent
from dataset_supervised_fonts import return_data
from PIL import Image
import torch.nn as nn
import functools
import networks
from torchvision import transforms


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    combine_sup_loss=[],
                    combine_unsup_loss=[],
                    images=[],
                    combine_supimages=[],
                    combine_unsupimages=[])

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        # self.global_iter = 0

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        # model params
        self.c_dim = args.c_dim
        self.image_size = args.image_size
        self.g_conv_dim = args.g_conv_dim
        self.g_repeat_num = args.g_repeat_num
        self.d_conv_dim = args.d_conv_dim
        self.d_repeat_num = args.d_repeat_num
        self.norm_layer = get_norm_layer(norm_type=args.norm)
        '''arrangement for each domain'''
        self.z_content_dim = args.z_content
        self.z_size_dim = args.z_size
        self.z_font_color_dim = args.z_font_color
        self.z_back_color_dim = args.z_back_color
        self.z_style_dim = args.z_style

        self.z_content_start_dim = 0
        self.z_size_start_dim = 20
        self.z_font_color_start_dim = 40
        self.z_back_color_start_dim = 60
        self.z_style_start_dim = 80


        self.lambda_combine = args.lambda_combine
        self.lambda_unsup = args.lambda_unsup



        if args.dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'ilab_unsup':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'ilab_sup':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'ilab_unsup_unbalance':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'ilab_unsup_unbalance_free':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'ilab_unsup_threeswap':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'fonts_unsup_nswap':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model
        # self.Autoencoder = Generator(self.nc, self.g_conv_dim, self.g_repeat_num)
        self.Autoencoder = Generator_fc(self.nc, self.g_conv_dim, self.g_repeat_num, self.z_dim)
        # self.Autoencoder = BetaVAE_ilab(self.z_dim, self.nc)

        self.Autoencoder.to(self.device)


        ''' use D '''
        # self.netD = networks.define_D(self.nc, self.d_conv_dim, 'basic',
        #                                 3, 'instance', True, 'normal', 0.02,
        #                                 '0,1')
        self.bg_classifier = Classifier_Latent(20, 10)
        self.bg_classifier.to(self.device)
        self.fg_classifier = Classifier_Latent(20, 10)
        self.fg_classifier.to(self.device)
        self.size_classifier = Classifier_Latent(20, 3)
        self.size_classifier.to(self.device)
        self.style_classifier = Classifier_Latent(20, 100)
        self.style_classifier.to(self.device)
        self.letter_classifier = Classifier_Latent(20, 52)
        self.letter_classifier.to(self.device)
        self.auto_optim = optim.Adam(list(self.Autoencoder.parameters()) + list(self.bg_classifier.parameters()) + list(self.fg_classifier.parameters()) +
                                      list(self.size_classifier.parameters()) + list(self.style_classifier.parameters())
                                     + list(self.letter_classifier.parameters()), lr=self.lr,
                                     betas=(self.beta1, self.beta2))
        # log
        self.log_dir = './checkpoints/' + args.viz_name
        self.model_save_dir = args.model_save_dir


        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_combine_sup = None
        self.win_combine_unsup = None
        # self.win_d_no_pose_losdata_loaders = None
        # self.win_d_pose_loss = None
        # self.win_equal_pose_loss = None
        # self.win_have_pose_loss = None
        # self.win_auto_loss_fake = None
        # self.win_loss_cor_coe = None
        # self.win_d_loss = None

        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)
        self.resume_iters = args.resume_iters

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        # if self.ckpt_name is not None:
        #     self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        self.gather = DataGather()
    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        Auto_path = os.path.join(self.model_save_dir, self.viz_name, '{}-Auto.ckpt'.format(resume_iters))
        self.Autoencoder.load_state_dict(torch.load(Auto_path, map_location=lambda storage, loc: storage))
        print("=> loaded checkpoint '{} (iter {})'".format(self.viz_name, resume_iters))

    def Cor_CoeLoss(self, y_pred, y_target):
        x = y_pred
        y = y_target
        x_var = x - torch.mean(x)
        y_var = y - torch.mean(y)
        r_num = torch.sum(x_var * y_var)
        r_den = torch.sqrt(torch.sum(x_var ** 2)) * torch.sqrt(torch.sum(y_var ** 2))
        r = r_num / r_den

        # return 1 - r  # best are 0
        return 1 - r ** 2  # abslute constrain

    def train(self):
        # self.net_mode(train=True)
        out = False
        # Start training from scratch or resume training.
        self.global_iter = 0
        if self.resume_iters:
            self.global_iter = self.resume_iters
            self.restore_model(self.resume_iters)

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        letter_dict = {}
        size_dict = {}
        fg_dict = {}
        bg_dict = {}
        style_dict = {}
        letter_cnt = 0
        size_cnt = 0
        fg_cnt = 0
        bg_cnt = 0
        style_cnt = 0
        for roots, dirs, files in os.walk('/home2/fonts_dataset_center'):
            for file in files:
                letter = file.split('_')[0]
                size = file.split('_')[1]
                fg = file.split('_')[2]
                bg = file.split('_')[3]
                style = file.split('_')[4].split('.')[0]
                # print(file)
                if letter not in letter_dict:
                    letter_dict[letter] = letter_cnt
                    letter_cnt += 1
                if size not in size_dict:
                    size_dict[size] = size_cnt
                    size_cnt += 1
                if fg not in fg:
                    fg_dict[fg] = fg_cnt
                    fg_cnt += 1
                if bg not in bg_dict:
                    bg_dict[bg] = bg_cnt
                    bg_cnt += 1
                if style not in style_dict:
                    style_dict[style] = style_cnt
                    style_cnt += 1
        while not out:
            for sup_package in self.data_loader:
                # appe, pose, combine
                A_img = sup_package['A']
                B_img = sup_package['B']
                C_img = sup_package['C']
                D_img = sup_package['D']
                E_img = sup_package['E']
                F_img = sup_package['F']
                self.global_iter += 1
                pbar.update(1)

                A_img = Variable(cuda(A_img, self.use_cuda))
                B_img = Variable(cuda(B_img, self.use_cuda))
                C_img = Variable(cuda(C_img, self.use_cuda))
                D_img = Variable(cuda(D_img, self.use_cuda))
                E_img = Variable(cuda(E_img, self.use_cuda))
                F_img = Variable(cuda(F_img, self.use_cuda))

                ## 1. A B C seperate(first400: id last600 background)
                A_recon, A_z = self.Autoencoder(A_img)
                B_recon, B_z = self.Autoencoder(B_img)
                C_recon, C_z = self.Autoencoder(C_img)
                D_recon, D_z = self.Autoencoder(D_img)
                E_recon, E_z = self.Autoencoder(E_img)
                F_recon, F_z = self.Autoencoder(F_img)
                ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''

                A_z_1 = A_z[:, 0:self.z_size_start_dim] # 0-200
                A_z_2 = A_z[:, self.z_size_start_dim : self.z_font_color_start_dim] # 20-40
                A_z_3 = A_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim] #40-60
                A_z_4 = A_z[:, self.z_back_color_start_dim : self.z_style_start_dim] # 60-80
                A_z_5 = A_z[:, self.z_style_start_dim :] #80-100
                B_z_1 = B_z[:, 0:self.z_size_start_dim] # 0-200
                B_z_2 = B_z[:, self.z_size_start_dim : self.z_font_color_start_dim] # 200-400
                B_z_3 = B_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim] #400-600
                B_z_4 = B_z[:, self.z_back_color_start_dim : self.z_style_start_dim] # 600-800
                B_z_5 = B_z[:, self.z_style_start_dim :] #800-1000
                C_z_1 = C_z[:, 0:self.z_size_start_dim] # 0-200
                C_z_2 = C_z[:, self.z_size_start_dim : self.z_font_color_start_dim] # 200-400
                C_z_3 = C_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim] #400-600
                C_z_4 = C_z[:, self.z_back_color_start_dim : self.z_style_start_dim] # 600-800
                C_z_5 = C_z[:, self.z_style_start_dim :] #800-1000
                D_z_1 = D_z[:, 0:self.z_size_start_dim] # 0-200
                D_z_2 = D_z[:, self.z_size_start_dim : self.z_font_color_start_dim] # 200-400
                D_z_3 = D_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim] #400-600
                D_z_4 = D_z[:, self.z_back_color_start_dim : self.z_style_start_dim] # 600-800
                D_z_5 = D_z[:, self.z_style_start_dim :] #800-1000
                E_z_1 = E_z[:, 0:self.z_size_start_dim] # 0-200
                E_z_2 = E_z[:, self.z_size_start_dim : self.z_font_color_start_dim] # 200-400
                E_z_3 = E_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim] #400-600
                E_z_4 = E_z[:, self.z_back_color_start_dim : self.z_style_start_dim] # 600-800
                E_z_5 = E_z[:, self.z_style_start_dim :] #800-1000
                F_z_1 = F_z[:, 0:self.z_size_start_dim] # 0-200
                F_z_2 = F_z[:, self.z_size_start_dim : self.z_font_color_start_dim] # 200-400
                F_z_3 = F_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim] #400-600
                F_z_4 = F_z[:, self.z_back_color_start_dim : self.z_style_start_dim] # 600-800
                F_z_5 = F_z[:, self.z_style_start_dim :] #800-1000

                ## 2. combine with strong supervise
                ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''
                # C A same content-1
                A1Co_combine_2C = torch.cat((A_z_1, C_z_2, C_z_3, C_z_4, C_z_5), dim=1)
                mid_A1Co = self.Autoencoder.fc_decoder(A1Co_combine_2C)
                mid_A1Co = mid_A1Co.view(A1Co_combine_2C.shape[0], 256, 8, 8)
                A1Co_2C = self.Autoencoder.decoder(mid_A1Co)

                AoC1_combine_2A = torch.cat((C_z_1, A_z_2, A_z_3, A_z_4, A_z_5), dim=1)
                mid_AoC1 = self.Autoencoder.fc_decoder(AoC1_combine_2A)
                mid_AoC1 = mid_AoC1.view(AoC1_combine_2A.shape[0], 256, 8, 8)
                AoC1_2A = self.Autoencoder.decoder(mid_AoC1)

                # C B same size 2
                B2Co_combine_2C = torch.cat((C_z_1, B_z_2, C_z_3, C_z_4, C_z_5), dim=1)
                mid_B2Co = self.Autoencoder.fc_decoder(B2Co_combine_2C)
                mid_B2Co = mid_B2Co.view(B2Co_combine_2C.shape[0], 256, 8, 8)
                B2Co_2C = self.Autoencoder.decoder(mid_B2Co)

                BoC2_combine_2B = torch.cat((B_z_1, C_z_2, B_z_3, B_z_4, B_z_5), dim=1)
                mid_BoC2 = self.Autoencoder.fc_decoder(BoC2_combine_2B)
                mid_BoC2 = mid_BoC2.view(BoC2_combine_2B.shape[0], 256, 8, 8)
                BoC2_2B = self.Autoencoder.decoder(mid_BoC2)

                # C D same font_color 3
                D3Co_combine_2C = torch.cat((C_z_1, C_z_2, D_z_3, C_z_4, C_z_5), dim=1)
                mid_D3Co = self.Autoencoder.fc_decoder(D3Co_combine_2C)
                mid_D3Co = mid_D3Co.view(D3Co_combine_2C.shape[0], 256, 8, 8)
                D3Co_2C = self.Autoencoder.decoder(mid_D3Co)

                DoC3_combine_2D = torch.cat((D_z_1, D_z_2, C_z_3, D_z_4, D_z_5), dim=1)
                mid_DoC3 = self.Autoencoder.fc_decoder(DoC3_combine_2D)
                mid_DoC3 = mid_DoC3.view(DoC3_combine_2D.shape[0], 256, 8, 8)
                DoC3_2D = self.Autoencoder.decoder(mid_DoC3)

                # C E same back_color 4
                E4Co_combine_2C = torch.cat((C_z_1, C_z_2, C_z_3, E_z_4, C_z_5), dim=1)
                mid_E4Co = self.Autoencoder.fc_decoder(E4Co_combine_2C)
                mid_E4Co = mid_E4Co.view(E4Co_combine_2C.shape[0], 256, 8, 8)
                E4Co_2C = self.Autoencoder.decoder(mid_E4Co)

                EoC4_combine_2E = torch.cat((E_z_1, E_z_2, E_z_3, C_z_4, E_z_5), dim=1)
                mid_EoC4 = self.Autoencoder.fc_decoder(EoC4_combine_2E)
                mid_EoC4 = mid_EoC4.view(EoC4_combine_2E.shape[0], 256, 8, 8)
                EoC4_2E = self.Autoencoder.decoder(mid_EoC4)

                # C F same style 5
                F5Co_combine_2C = torch.cat((C_z_1, C_z_2, C_z_3, C_z_4, F_z_5), dim=1)
                mid_F5Co = self.Autoencoder.fc_decoder(F5Co_combine_2C)
                mid_F5Co = mid_F5Co.view(F5Co_combine_2C.shape[0], 256, 8, 8)
                F5Co_2C = self.Autoencoder.decoder(mid_F5Co)

                FoC5_combine_2F = torch.cat((F_z_1, F_z_2, F_z_3, F_z_4, C_z_5), dim=1)
                mid_FoC5 = self.Autoencoder.fc_decoder(FoC5_combine_2F)
                mid_FoC5 = mid_FoC5.view(FoC5_combine_2F.shape[0], 256, 8, 8)
                FoC5_2F = self.Autoencoder.decoder(mid_FoC5)


                # combine_2C
                A1B2D3E4F5_combine_2C = torch.cat((A_z_1, B_z_2, D_z_3, E_z_4, F_z_5), dim=1)
                mid_A1B2D3E4F5 = self.Autoencoder.fc_decoder(A1B2D3E4F5_combine_2C)
                mid_A1B2D3E4F5 = mid_A1B2D3E4F5.view(A1B2D3E4F5_combine_2C.shape[0], 256, 8, 8)
                A1B2D3E4F5_2C = self.Autoencoder.decoder(mid_A1B2D3E4F5)



                # '''  need unsupervise '''
                A2B3D4E5F1_combine_2N = torch.cat((F_z_1, A_z_2, B_z_3, D_z_4, E_z_5), dim=1)
                mid_A2B3D4E5F1 = self.Autoencoder.fc_decoder(A2B3D4E5F1_combine_2N)
                mid_A2B3D4E5F1 = mid_A2B3D4E5F1.view(A2B3D4E5F1_combine_2N.shape[0], 256, 8, 8)
                A2B3D4E5F1_2N = self.Autoencoder.decoder(mid_A2B3D4E5F1)


                '''
                optimize for autoencoder
                '''

                # 1. recon_loss
                A_recon_loss = torch.mean(torch.abs(A_img - A_recon))
                B_recon_loss = torch.mean(torch.abs(B_img - B_recon))
                C_recon_loss = torch.mean(torch.abs(C_img - C_recon))
                D_recon_loss = torch.mean(torch.abs(D_img - D_recon))
                E_recon_loss = torch.mean(torch.abs(E_img - E_recon))
                F_recon_loss = torch.mean(torch.abs(F_img - F_recon))
                recon_loss = A_recon_loss + B_recon_loss + C_recon_loss + D_recon_loss + E_recon_loss + F_recon_loss

                # 2. sup_combine_loss
                A1Co_2C_loss = torch.mean(torch.abs(C_img - A1Co_2C))
                AoC1_2A_loss = torch.mean(torch.abs(A_img - AoC1_2A))
                B2Co_2C_loss = torch.mean(torch.abs(C_img - B2Co_2C))
                BoC2_2B_loss = torch.mean(torch.abs(B_img - BoC2_2B))
                D3Co_2C_loss = torch.mean(torch.abs(C_img - D3Co_2C))
                DoC3_2D_loss = torch.mean(torch.abs(D_img - DoC3_2D))
                E4Co_2C_loss = torch.mean(torch.abs(C_img - E4Co_2C))
                EoC4_2E_loss = torch.mean(torch.abs(E_img - EoC4_2E))
                F5Co_2C_loss = torch.mean(torch.abs(C_img - F5Co_2C))
                FoC5_2F_loss = torch.mean(torch.abs(F_img - FoC5_2F))
                A1B2D3E4F5_2C_loss = torch.mean(torch.abs(C_img - A1B2D3E4F5_2C))
                combine_sup_loss = A1Co_2C_loss + AoC1_2A_loss + B2Co_2C_loss + BoC2_2B_loss + D3Co_2C_loss + DoC3_2D_loss + E4Co_2C_loss + EoC4_2E_loss + F5Co_2C_loss + FoC5_2F_loss + A1B2D3E4F5_2C_loss

                # 3. unsup_combine_loss
                _, A2B3D4E5F1_z = self.Autoencoder(A2B3D4E5F1_2N)
                combine_unsup_loss = torch.mean(torch.abs(F_z_1 - A2B3D4E5F1_z[:, 0:self.z_size_start_dim])) + torch.mean(torch.abs(A_z_2 - A2B3D4E5F1_z[:, self.z_size_start_dim : self.z_font_color_start_dim])) \
                                     + torch.mean(torch.abs(B_z_3 - A2B3D4E5F1_z[:, self.z_font_color_start_dim : self.z_back_color_start_dim])) \
                                     + torch.mean(torch.abs(D_z_4 - A2B3D4E5F1_z[:, self.z_back_color_start_dim : self.z_style_start_dim])) \
                                     + torch.mean(torch.abs(E_z_5 - A2B3D4E5F1_z[:, self.z_style_start_dim :]))
                #  4. classification_loss
                labels = sup_package['labels']
                classification_criterion = nn.CrossEntropyLoss()
                classification_loss = 0
                letter_list = [A_z_1, B_z_1, C_z_1, D_z_1, E_z_1, F_z_1]
                size_list = [A_z_2, B_z_2, C_z_2, D_z_2, E_z_2, F_z_2]
                fg_list = [A_z_3, B_z_3, C_z_3, D_z_3, E_z_3, F_z_3]
                bg_list = [A_z_4, B_z_4, C_z_4, D_z_4, E_z_4, F_z_4]
                style_list = [A_z_5, B_z_5, C_z_5, D_z_5, E_z_5, F_z_5]
                print()
                for i in range(6):
                    classification_loss += classification_criterion(self.letter_classifier(letter_list[i]), torch.tensor(labels['letter'][i], device=self.device).long())
                    classification_loss += classification_criterion(self.size_classifier(size_list[i]),
                                                                    torch.tensor(labels['size'][i], device=self.device).long())
                    classification_loss += classification_criterion(self.fg_classifier(fg_list[i]),
                                                                    torch.tensor(labels['fg'][i], device=self.device).long())
                    classification_loss += classification_criterion(self.bg_classifier(bg_list[i]),
                                                                    torch.tensor(labels['bg'][i], device=self.device).long())
                    classification_loss += classification_criterion(self.style_classifier(style_list[i]),
                                                                    torch.tensor(labels['style'][i], device=self.device).long())
                # whole loss
                vae_unsup_loss = recon_loss + classification_loss#+ self.lambda_combine * combine_sup_loss + self.lambda_unsup * combine_unsup_loss
                self.auto_optim.zero_grad()
                vae_unsup_loss.backward()
                self.auto_optim.step()

                #　save the log
                f = open(self.log_dir + '/log.txt', 'a')
                f.writelines(['\n', '[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f} classification_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data, classification_loss.data)])
                f.close()
                print(['\n', '[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f} classification_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data, classification_loss.data)])

                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,recon_loss=recon_loss.data,
                                    combine_sup_loss=combine_sup_loss.data, combine_unsup_loss=combine_unsup_loss.data)

                if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f}  combine_sup_loss:{:.3f}  combine_unsup_loss:{:.3f} classification_loss:{:.3f}'.format(
                        self.global_iter, recon_loss.data, combine_sup_loss.data, combine_unsup_loss.data, classification_loss.data))

                    if self.viz_on:
                        self.gather.insert(images=A_img.data)
                        self.gather.insert(images=B_img.data)
                        self.gather.insert(images=C_img.data)
                        self.gather.insert(images=D_img.data)
                        self.gather.insert(images=E_img.data)
                        self.gather.insert(images=F_img.data)
                        self.gather.insert(images=F.sigmoid(A_recon).data)
                        self.viz_reconstruction()
                        self.viz_lines()
                        '''
                        combine show
                        '''
                        self.gather.insert(combine_supimages=F.sigmoid(AoC1_2A).data)
                        self.gather.insert(combine_supimages=F.sigmoid(BoC2_2B).data)
                        self.gather.insert(combine_supimages=F.sigmoid(D3Co_2C).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DoC3_2D).data)
                        self.gather.insert(combine_supimages=F.sigmoid(EoC4_2E).data)
                        self.gather.insert(combine_supimages=F.sigmoid(FoC5_2F).data)
                        self.viz_combine_recon()

                        self.gather.insert(combine_unsupimages=F.sigmoid(A1B2D3E4F5_2C).data)
                        self.gather.insert(combine_unsupimages=F.sigmoid(A2B3D4E5F1_2N).data)
                        self.viz_combine_unsuprecon()
                        # self.viz_combine(x)
                        self.gather.flush()
                # Save model checkpoints.
                if self.global_iter%self.save_step == 0:
                    Auto_path = os.path.join(self.model_save_dir, self.viz_name, '{}-Auto.ckpt'.format(self.global_iter))
                    torch.save(self.Autoencoder.state_dict(), Auto_path)
                    print('Saved model checkpoints into {}/{}...'.format(self.model_save_dir, self.viz_name))


                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
    def save_sample_img(self, tensor, mode):
        unloader = transforms.ToPILImage()
        dir = os.path.join(self.model_save_dir, self.viz_name, 'sample_img')
        if not os.path.exists(dir):
            os.makedirs(dir)
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it

        if mode == 'recon':
            image_ori_A = image[0].squeeze(0)  # remove the fake batch dimension
            image_ori_B = image[1].squeeze(0)
            image_ori_C = image[2].squeeze(0)
            image_ori_D = image[3].squeeze(0)
            image_ori_E = image[4].squeeze(0)
            image_ori_F = image[5].squeeze(0)
            image_recon = image[6].squeeze(0)

            image_ori_A = unloader(image_ori_A)
            image_ori_B = unloader(image_ori_B)
            image_ori_C = unloader(image_ori_C)
            image_ori_D = unloader(image_ori_D)
            image_ori_E = unloader(image_ori_E)
            image_ori_F = unloader(image_ori_F)
            image_recon = unloader(image_recon)

            image_ori_A.save(os.path.join(dir, '{}-A_img.png'.format(self.global_iter)))
            image_ori_B.save(os.path.join(dir, '{}-B_img.png'.format(self.global_iter)))
            image_ori_C.save(os.path.join(dir, '{}-C_img.png'.format(self.global_iter)))
            image_ori_D.save(os.path.join(dir, '{}-D_img.png'.format(self.global_iter)))
            image_ori_E.save(os.path.join(dir, '{}-E_img.png'.format(self.global_iter)))
            image_ori_F.save(os.path.join(dir, '{}-F_img.png'.format(self.global_iter)))
            image_recon.save(os.path.join(dir, '{}-A_img_recon.png'.format(self.global_iter)))
        elif mode == 'combine_sup':

            image_AoC1_2A = image[0].squeeze(0)  # remove the fake batch dimension
            image_BoC2_2B = image[1].squeeze(0)
            image_D3Co_2C = image[2].squeeze(0)
            image_DoC3_2D = image[3].squeeze(0)
            image_EoC4_2E = image[4].squeeze(0)
            image_FoC5_2F = image[5].squeeze(0)

            image_AoC1_2A = unloader(image_AoC1_2A)
            image_BoC2_2B = unloader(image_BoC2_2B)
            image_D3Co_2C = unloader(image_D3Co_2C)
            image_DoC3_2D = unloader(image_DoC3_2D)
            image_EoC4_2E = unloader(image_EoC4_2E)
            image_FoC5_2F = unloader(image_FoC5_2F)

            image_AoC1_2A.save(os.path.join(dir, '{}-AoC1_2A.png'.format(self.global_iter)))
            image_BoC2_2B.save(os.path.join(dir, '{}-BoC2_2B.png'.format(self.global_iter)))
            image_D3Co_2C.save(os.path.join(dir, '{}-D3Co_2C.png'.format(self.global_iter)))
            image_DoC3_2D.save(os.path.join(dir, '{}-DoC3_2D.png'.format(self.global_iter)))
            image_EoC4_2E.save(os.path.join(dir, '{}-EoC4_2E.png'.format(self.global_iter)))
            image_FoC5_2F.save(os.path.join(dir, '{}-FoC5_2F.png'.format(self.global_iter)))

        elif mode == 'combine_unsup':
            image_A1B2D3E4F5_2C = image[0].squeeze(0)  # remove the fake batch dimension
            image_A2B3D4E5F1_2N = image[1].squeeze(0)

            image_A1B2D3E4F5_2C = unloader(image_A1B2D3E4F5_2C)
            image_A2B3D4E5F1_2N = unloader(image_A2B3D4E5F1_2N)

            image_A1B2D3E4F5_2C.save(os.path.join(dir, '{}-A1B2D3E4F5_2C.png'.format(self.global_iter)))
            image_A2B3D4E5F1_2N.save(os.path.join(dir, '{}-A2B3D4E5F1_2N.png'.format(self.global_iter)))
    def viz_reconstruction(self):
        # self.net_mode(train=False)
        x_A = self.gather.data['images'][0][:100]
        x_A = make_grid(x_A, normalize=True)
        x_B = self.gather.data['images'][1][:100]
        x_B = make_grid(x_B, normalize=True)
        x_C = self.gather.data['images'][2][:100]
        x_C = make_grid(x_C, normalize=True)
        x_D = self.gather.data['images'][3][:100]
        x_D = make_grid(x_D, normalize=True)
        x_E = self.gather.data['images'][4][:100]
        x_E = make_grid(x_E, normalize=True)
        x_F = self.gather.data['images'][5][:100]
        x_F = make_grid(x_F, normalize=True)
        x_A_recon = self.gather.data['images'][6][:100]
        x_A_recon = make_grid(x_A_recon, normalize=True)
        images = torch.stack([x_A, x_B, x_C, x_D, x_E, x_F, x_A_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.save_sample_img(images, 'recon')
        # self.net_mode(train=True)
    def viz_combine_recon(self):
        # self.net_mode(train=False)
        AoC1_2A = self.gather.data['combine_supimages'][0][:100]
        AoC1_2A = make_grid(AoC1_2A, normalize=True)
        BoC2_2B = self.gather.data['combine_supimages'][1][:100]
        BoC2_2B = make_grid(BoC2_2B, normalize=True)
        D3Co_2C = self.gather.data['combine_supimages'][2][:100]
        D3Co_2C = make_grid(D3Co_2C, normalize=True)
        DoC3_2D = self.gather.data['combine_supimages'][3][:100]
        DoC3_2D = make_grid(DoC3_2D, normalize=True)
        EoC4_2E = self.gather.data['combine_supimages'][4][:100]
        EoC4_2E = make_grid(EoC4_2E, normalize=True)
        FoC5_2F = self.gather.data['combine_supimages'][5][:100]
        FoC5_2F = make_grid(FoC5_2F, normalize=True)
        images = torch.stack([AoC1_2A, BoC2_2B, D3Co_2C, DoC3_2D, EoC4_2E, FoC5_2F], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'combine_supimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.save_sample_img(images, 'combine_sup')
    def viz_combine_unsuprecon(self):
        # self.net_mode(train=False)
        A1B2D3E4F5_2C = self.gather.data['combine_unsupimages'][0][:100]
        A1B2D3E4F5_2C = make_grid(A1B2D3E4F5_2C, normalize=True)
        A2B3D4E5F1_2N = self.gather.data['combine_unsupimages'][1][:100]
        A2B3D4E5F1_2N = make_grid(A2B3D4E5F1_2N, normalize=True)
        images = torch.stack([A1B2D3E4F5_2C, A2B3D4E5F1_2N], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'combine_unsupimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.save_sample_img(images, 'combine_unsup')


    def viz_combine(self, x):
        # self.net_mode(train=False)

        decoder = self.Autoencoder.decoder
        encoder = self.Autoencoder.encoder
        z = encoder(x)
        z_appe = z[:, 0:250, :, :]
        z_pose = z[:, 250:, :, :]
        z_rearrange_combine = torch.cat((z_appe[:-1], z_pose[1:]), dim=1)
        x_rearrange_combine = decoder(z_rearrange_combine)
        x_rearrange_combine = F.sigmoid(x_rearrange_combine).data

        x_show = make_grid(x[:-1].data, normalize=True)
        x_rearrange_combine_show = make_grid(x_rearrange_combine, normalize=True)
        images = torch.stack([x_show, x_rearrange_combine_show], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_combine',
                        opts=dict(title=str(self.global_iter)), nrow=10)




        # samples = []
        # for i in range(10): # every pair need visualize
        #     x_appe = x[i].unsqueeze(0)  # provide appearance
        #
        #     z_appe = z[i, 0:250, :, :].unsqueeze(0)  # provide appearance
        #     x_pose = x[i+1].unsqueeze(0)  # provide pose
        #
        #     z_pose = z[i+1, 250:, :, :].unsqueeze(0)  # provide pose
        #     z_combine = torch.cat((z_appe, z_pose), 1)
        #     x_combine = decoder(z_combine)
        #     x_combine = F.sigmoid(x_combine).data
        #     samples.append(x_appe)
        #     samples.append(x_combine)
        #     samples.append(x_pose)
        #     samples = torch.cat(samples, dim=0).cpu()
        #     title = 'combine(iter:{})'.format(self.global_iter)
        #     if self.viz_on:
        #         self.viz.images(samples, env=self.viz_name+'combine',
        #                         opts=dict(title=title))

    def viz_lines(self):
        # self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()

        combine_sup_loss = torch.stack(self.gather.data['combine_sup_loss']).cpu()
        combine_unsup_loss = torch.stack(self.gather.data['combine_unsup_loss']).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))
        legend.append('mean')
        legend.append('total')

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))

        if self.win_combine_sup is None:
            self.win_combine_sup = self.viz.line(
                                        X=iters,
                                        Y=combine_sup_loss,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='combine_sup_loss',))
        else:
            self.win_combine_sup = self.viz.line(
                                        X=iters,
                                        Y=combine_sup_loss,
                                        env=self.viz_name+'_lines',
                                        win=self.win_combine_sup,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='combine_sup_loss',))

        if self.win_combine_unsup is None:
            self.win_combine_unsup = self.viz.line(
                                        X=iters,
                                        Y=combine_unsup_loss,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='combine_unsup_loss',))
        else:
            self.win_combine_unsup = self.viz.line(
                                        X=iters,
                                        Y=combine_sup_loss,
                                        env=self.viz_name+'_lines',
                                        win=self.win_combine_unsup,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='combine_unsup_loss',))


    def viz_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = Variable(cuda(fixed_img1, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = Variable(cuda(fixed_img2, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)

            if self.viz_on:
                self.viz.images(samples, env=self.viz_name+'_traverse',
                                opts=dict(title=title), nrow=len(interpolation))

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
