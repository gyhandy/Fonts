"""solver.py"""

import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import visdom
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif
from model_checkz import BetaVAE_H, BetaVAE_B
# from dataset_Fonts_swap import return_data
# from dataset import return_data
# from dataset_Nswap_dsprites import return_data
from dataset_Nswap_dsprites_swap import return_data
from model_sup_changeZdim import Generator_fc
from torchvision import transforms
import functools


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
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],combine_sup_loss=[],
                    combine_unsup_loss=[],
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
        self.global_iter = 0
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
        self.model_save_dir = args.model_save_dir

        self.nc = 3
        self.decoder_dist = 'gaussian'

        self.z_content_start_dim = 0
        self.z_size_start_dim = 2
        self.z_font_color_start_dim = 4
        self.z_back_color_start_dim = 6
        self.z_style_start_dim = 8

        self.lambda_combine = args.lambda_combine
        self.lambda_unsup = args.lambda_unsup

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.Autoencoder = Generator_fc(self.nc, self.g_conv_dim, self.g_repeat_num, self.z_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Autoencoder.to(self.device)
        self.resume_iters = args.resume_iters
        if self.resume_iters:
            self.global_iter = self.resume_iters
            self.restore_model(self.resume_iters)

        if args.dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        if args.dataset.lower() == 'dsprites_unsup_nswap':
            self.nc = 1
            self.decoder_dist = 'bernoulli'

        elif args.dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'fonts':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError

        if args.model == 'H':
            net = BetaVAE_H
        elif args.model == 'B':
            net = BetaVAE_B

        else:
            raise NotImplementedError('only support model H or B')

        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        self.log_dir = './checkpoints/' + args.viz_name
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

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

    def train(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            for sup_package in self.data_loader:
                x_name = '..'
                self.global_iter += 1
                pbar.update(1)

                # x =
                # x = Variable(cuda(x, self.use_cuda))
                # x_recon, mu, logvar, z = self.net(x)
                #
                # recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                # total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                unloader = transforms.ToPILImage()

                # for item in range(len(x)):
                #     z_save = str(z.data.cpu().tolist()[item])
                #     f = open(self.log_dir + '/name_z.txt', 'a')
                #     f.writelines([x_name[item]+','+z_save[1:-1] + '\n'])
                #     f.close()
                #
                #     dir = os.path.join(self.log_dir, 'sample_img_checkz')
                #     for i in range(100):
                #         # print(i)
                #         os.makedirs(dir+'/%d'%i, exist_ok=True)
                #         for j in [300, 0.01, -300, 1]:
                #             A_z_temp = z.clone()
                #             A_z_temp[item][i] *= j
                #
                #
                #             img_Az = self.net.decoder(A_z_temp)
                #
                #             pre_img_Az = F.sigmoid(img_Az).data.squeeze(0).cpu()
                #             image_Az = unloader(pre_img_Az)
                #             image_Az.save(os.path.join(dir+'/%d'%i, '{}-{}-{}_img.png'.format(x_name[item], j, i)))

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


                A_recon_vae, A_mu, A_logvar, A_z_vae = self.net(A_img)
                B_recon_vae, B_mu, B_logvar, B_z_vae = self.net(B_img)
                C_recon_vae, C_mu, C_logvar, C_z_vae = self.net(C_img)
                D_recon_vae, D_mu, D_logvar, D_z_vae = self.net(D_img)
                E_recon_vae, E_mu, E_logvar, E_z_vae = self.net(E_img)
                F_recon_vae, F_mu, F_logvar, F_z_vae = self.net(F_img)
                # dir = os.path.join(self.log_dir)
                # image_A_recon_vae = []
                # image_B_recon_vae = []
                # image_C_recon_vae = []
                # image_D_recon_vae = []
                # image_E_recon_vae = []
                # image_F_recon_vae = []
                # for item in range(len(A_recon_vae)):
                #     image_A_recon_vae.append(unloader(F.sigmoid(A_recon_vae[item].data.cpu())))
                #
                #     image_A_recon_vae[item].save(os.path.join(dir+'/sample_img', '{}-{}-{}_img.png'.format(self.global_iter, item, 'A_recon_vae')))
                #     image_B_recon_vae.append(unloader(F.sigmoid(B_recon_vae[item].data.cpu())))
                #     image_B_recon_vae[item].save(os.path.join(dir + '/sample_img',
                #                                         '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                   'B_recon_vae')))
                #     image_C_recon_vae.append(unloader(F.sigmoid(C_recon_vae[item].data.cpu())))
                #     image_C_recon_vae[item].save(os.path.join(dir + '/sample_img',
                #                                         '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                   'C_recon_vae')))
                #     image_D_recon_vae.append(unloader(F.sigmoid(D_recon_vae[item].data.cpu())))
                #     image_D_recon_vae[item].save(os.path.join(dir + '/sample_img',
                #                                         '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                   'D_recon_vae')))
                #     image_E_recon_vae.append(unloader(F.sigmoid(E_recon_vae[item].data.cpu())))
                #     image_E_recon_vae[item].save(os.path.join(dir + '/sample_img',
                #                                         '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                   'E_recon_vae')))
                #     image_F_recon_vae.append(unloader(F.sigmoid(F_recon_vae[item].data.cpu())))
                #     image_F_recon_vae[item].save(os.path.join(dir + '/sample_img',
                #                                         '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                   'F_recon_vae')))

                # for item in range(len(A_img)):
                #     image_A_img = unloader(A_img[item].cpu())
                #     image_A_img.save(os.path.join(dir+'/sample_img', '{}-{}-{}_img.png'.format(self.global_iter, item, 'A_img')))
                #     image_B_img = unloader(B_img[item].cpu())
                #     image_B_img.save(
                #         os.path.join(dir + '/sample_img', '{}-{}-{}_img.png'.format(self.global_iter, item, 'B_img')))
                #     image_C_img = unloader(C_img[item].cpu())
                #     image_C_img.save(
                #         os.path.join(dir + '/sample_img', '{}-{}-{}_img.png'.format(self.global_iter, item, 'C_img')))
                #     image_D_img = unloader(D_img[item].cpu())
                #     image_D_img.save(
                #         os.path.join(dir + '/sample_img', '{}-{}-{}_img.png'.format(self.global_iter, item, 'D_img')))
                #     image_E_img = unloader(E_img[item].cpu())
                #     image_E_img.save(
                #         os.path.join(dir + '/sample_img', '{}-{}-{}_img.png'.format(self.global_iter, item, 'E_img')))
                #     image_F_img = unloader(F_img[item].cpu())
                #     image_F_img.save(
                #         os.path.join(dir + '/sample_img', '{}-{}-{}_img.png'.format(self.global_iter, item, 'F_img')))

                ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''
                #b-vae 33
                # new = [9, 16, 51, 81, 79, 68, 62, 66, 78, 93, 59, 94, 12, 61, 96, 0, 85, 45, 28, 87, 76, 20, 48, 57, 84,
                #        91,
                #        75, 6, 43, 80, 18, 53, 55, 38, 42, 34, 4, 10, 92, 71, 27, 8, 44, 63, 15, 65, 56, 90, 11, 3, 25,
                #        73, 83,
                #        21, 26, 72, 31, 47, 97, 36, 86, 52, 77, 49, 24, 70, 88, 50, 82, 41, 5, 39, 14, 35, 40, 67, 32,
                #        98, 74,
                #        19, 29, 37, 1, 89, 13, 95, 7, 17, 60, 30, 99, 54, 64, 69, 2, 58, 22, 46, 33, 23]


                #iso-vae 34
                # new = [66, 41, 87, 34, 95, 47, 63, 25, 55, 94, 21, 78, 20, 73, 49, 65, 61, 56, 88, 58, 31, 62, 5, 8, 33,
                #        69, 57, 42, 97, 24, 4, 38, 46, 23, 40, 96, 77, 48, 44, 7, 15, 22, 9, 2, 98, 0, 82, 91, 99, 6, 85,
                #        14, 83, 68, 64, 12, 72, 35, 81, 79, 75, 1, 51, 53, 36, 29, 50, 26, 71, 90, 17, 10, 39, 28, 67,
                #        76, 37, 80, 19, 11, 13, 84, 30, 86, 52, 27, 16, 45, 32, 74, 59, 70, 92, 54, 93, 43, 89, 60, 3,
                #        18]
                #
                # fg_idx = new[0:20]
                # bg_idx = new[20:40]
                # size_idx = new[40:60]
                # style_idx = new[60:80]
                # content_idx = new[80:100]

                # new = [8, 2, 7, 3, 0, 5, 1, 9, 4, 7] # 0~10 # vae
                # new = [5, 0, 9, 4, 7, 8, 2, 1, 3, 6] # 0~10 # tcvae
                # new = [5, 4, 6, 2, 7, 3, 8, 0, 9, 1]  # 0~10 # tcvae
                new = [8, 9, 6, 3, 0, 2, 1, 5, 4, 7]  # 0~10 # tcvae-vae

                # fg_idx = new[0:20]
                # bg_idx = new[20:40]
                # size_idx = new[40:60]
                # style_idx = new[60:80]
                # content_idx = new[80:100]
                content_idx = new[0:2] # 1 shape
                size_idx = new[2:4] # 2 scale
                fg_idx = new[4:6] # 3 orientation
                bg_idx = new[6:8] # 4 X
                style_idx = new[8:10] # 5 Y


                # fg_idx = new[0:20]
                # bg_idx = [14, 20, 21, 22, 29, 32, 51, 56, 57, 58, 60, 63, 72, 77, 80, 91, 95, 96, 97, 98]
                # size_idx = [93, 88, 76, 33]
                # style_idx = [37, 38, 39, 42, 66, 70, 79, 83, 90, 92, ]
                # content_idx = [0,3,4,5,6,7,8,9,10,11,13,15,16,17,18,19,24,28,30,34,35,36,40,41,43,44,45,46,47,48,49,50,\
                #                53,59,64,65,67,68,71,73,74,78,81,82,84,85,86,89,99]
                #





                A_z_1_vae = torch.tensor([[A_z_vae[0][i] for i in content_idx]]).to(self.device)  # 0-20
                A_z_2_vae = torch.tensor([[A_z_vae[0][i] for i in size_idx]]).to(self.device)  # 20-40
                A_z_3_vae = torch.tensor([[A_z_vae[0][i] for i in fg_idx]]).to(self.device)  # 40-60
                A_z_4_vae = torch.tensor([[A_z_vae[0][i] for i in bg_idx]]).to(self.device)  # 60-80
                A_z_5_vae = torch.tensor([[A_z_vae[0][i] for i in style_idx]]).to(self.device)  # 80-100
                B_z_1_vae = torch.tensor([[B_z_vae[0][i] for i in content_idx]]).to(self.device)  # 0-200
                B_z_2_vae = torch.tensor([[B_z_vae[0][i] for i in size_idx]]).to(self.device)  # 20-40
                B_z_3_vae = torch.tensor([[B_z_vae[0][i] for i in fg_idx]]).to(self.device)  # 40-60
                B_z_4_vae = torch.tensor([[B_z_vae[0][i] for i in bg_idx]]).to(self.device)  # 60-80
                B_z_5_vae = torch.tensor([[B_z_vae[0][i] for i in style_idx]]).to(self.device)  # 80-100
                C_z_1_vae = torch.tensor([[C_z_vae[0][i] for i in content_idx]]).to(self.device)  # 0-200
                C_z_2_vae = torch.tensor([[C_z_vae[0][i] for i in size_idx]]).to(self.device)  # 20-40
                C_z_3_vae = torch.tensor([[C_z_vae[0][i] for i in fg_idx]]).to(self.device)  # 40-60
                C_z_4_vae = torch.tensor([[C_z_vae[0][i] for i in bg_idx]]).to(self.device)  # 60-80
                C_z_5_vae = torch.tensor([[C_z_vae[0][i] for i in style_idx]]).to(self.device)  # 80-100
                D_z_1_vae = torch.tensor([[D_z_vae[0][i] for i in content_idx]]).to(self.device)  # 0-200
                D_z_2_vae = torch.tensor([[D_z_vae[0][i] for i in size_idx]]).to(self.device)  # 20-40
                D_z_3_vae = torch.tensor([[D_z_vae[0][i] for i in fg_idx]]).to(self.device)  # 40-60
                D_z_4_vae = torch.tensor([[D_z_vae[0][i] for i in bg_idx]]).to(self.device)  # 60-80
                D_z_5_vae = torch.tensor([[D_z_vae[0][i] for i in style_idx]]).to(self.device)  # 80-100
                E_z_1_vae = torch.tensor([[E_z_vae[0][i] for i in content_idx]]).to(self.device)  # 0-200
                E_z_2_vae = torch.tensor([[E_z_vae[0][i] for i in size_idx]]).to(self.device)  # 20-40
                E_z_3_vae = torch.tensor([[E_z_vae[0][i] for i in fg_idx]]).to(self.device)  # 40-60
                E_z_4_vae = torch.tensor([[E_z_vae[0][i] for i in bg_idx]]).to(self.device)  # 60-80
                E_z_5_vae = torch.tensor([[E_z_vae[0][i] for i in style_idx]]).to(self.device)  # 80-100
                F_z_1_vae = torch.tensor([[F_z_vae[0][i] for i in content_idx]]).to(self.device)  # 0-200
                F_z_2_vae = torch.tensor([[F_z_vae[0][i] for i in size_idx]]).to(self.device)  # 20-40
                F_z_3_vae = torch.tensor([[F_z_vae[0][i] for i in fg_idx]]).to(self.device)  # 40-60
                F_z_4_vae = torch.tensor([[F_z_vae[0][i] for i in bg_idx]]).to(self.device)  # 60-80
                F_z_5_vae = torch.tensor([[F_z_vae[0][i] for i in style_idx]]).to(self.device)  # 80-100

                # A_z_1_vae = A_z_vae[:, 0:self.z_size_start_dim]  # 0-200
                # A_z_2_vae = A_z_vae[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
                # A_z_3_vae = A_z_vae[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
                # A_z_4_vae = A_z_vae[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
                # A_z_5_vae = A_z_vae[:, self.z_style_start_dim:]  # 80-100
                # B_z_1_vae = B_z_vae[:, 0:self.z_size_start_dim]  # 0-200
                # B_z_2_vae = B_z_vae[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 200-400
                # B_z_3_vae = B_z_vae[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 400-600
                # B_z_4_vae = B_z_vae[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 600-800
                # B_z_5_vae = B_z_vae[:, self.z_style_start_dim:]  # 800-1000
                # C_z_1_vae = C_z_vae[:, 0:self.z_size_start_dim]  # 0-200
                # C_z_2_vae = C_z_vae[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 200-400
                # C_z_3_vae = C_z_vae[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 400-600
                # C_z_4_vae = C_z_vae[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 600-800
                # C_z_5_vae = C_z_vae[:, self.z_style_start_dim:]  # 800-1000
                # D_z_1_vae = D_z_vae[:, 0:self.z_size_start_dim]  # 0-200
                # D_z_2_vae = D_z_vae[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 200-400
                # D_z_3_vae = D_z_vae[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 400-600
                # D_z_4_vae = D_z_vae[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 600-800
                # D_z_5_vae = D_z_vae[:, self.z_style_start_dim:]  # 800-1000
                # E_z_1_vae = E_z_vae[:, 0:self.z_size_start_dim]  # 0-200
                # E_z_2_vae = E_z_vae[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 200-400
                # E_z_3_vae = E_z_vae[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 400-600
                # E_z_4_vae = E_z_vae[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 600-800
                # E_z_5_vae = E_z_vae[:, self.z_style_start_dim:]  # 800-1000
                # F_z_1_vae = F_z_vae[:, 0:self.z_size_start_dim]  # 0-200
                # F_z_2_vae = F_z_vae[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 200-400
                # F_z_3_vae = F_z_vae[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 400-600
                # F_z_4_vae = F_z_vae[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 600-800
                # F_z_5_vae = F_z_vae[:, self.z_style_start_dim:]  # 800-1000
                ## 2. combine with strong supervise
                ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''

                # C A same content-1
                A1Co_combine_2C_vae = torch.cat((A_z_1_vae, C_z_2_vae, C_z_3_vae, C_z_4_vae, C_z_5_vae), dim=1)
                A1Co_2C_vae = self.net.decoder(A1Co_combine_2C_vae)
                # pre_img_A1Co_combine_2C_vae = A1Co_2C_vae.data.squeeze(0).cpu()
                # image_A1Co_combine_2C_vae = []
                # for item in range(len(pre_img_A1Co_combine_2C_vae)):
                #     image_A1Co_combine_2C_vae.append(unloader(F.sigmoid(pre_img_A1Co_combine_2C_vae[item].data)))
                #     image_A1Co_combine_2C_vae[item].save(os.path.join(dir+'/sample_img', '{}-{}-{}_img.png'.format(self.global_iter, item, 'A1Co_combine_2C_vae')))
                #
                AoC1_combine_2A_vae = torch.cat((C_z_1_vae, A_z_2_vae, A_z_3_vae, A_z_4_vae, A_z_5_vae), dim=1)
                AoC1_2A_vae = self.net.decoder(AoC1_combine_2A_vae)
                # pre_img_AoC1_combine_2A_vae = AoC1_combine_2A_vae.data.squeeze(0).cpu()
                # image_AoC1_combine_2A_vae = []
                # for item in range(len(pre_img_AoC1_combine_2A_vae)):
                #     image_AoC1_combine_2A_vae.append(unloader(F.sigmoid(pre_img_AoC1_combine_2A_vae[item].data)))
                #     image_AoC1_combine_2A_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'AoC1_combine_2A_vae')))

                # C B same size 2
                B2Co_combine_2C_vae = torch.cat((C_z_1_vae, B_z_2_vae, C_z_3_vae, C_z_4_vae, C_z_5_vae), dim=1)
                B2Co_2C_vae = self.net.decoder(B2Co_combine_2C_vae)
                # pre_img_B2Co_combine_2C_vae = B2Co_combine_2C_vae.data.squeeze(0).cpu()
                # image_B2Co_combine_2C_vae = []
                # for item in range(len(pre_img_B2Co_combine_2C_vae)):
                #     image_B2Co_combine_2C_vae.append(unloader(F.sigmoid(pre_img_B2Co_combine_2C_vae[item].data)))
                #     image_B2Co_combine_2C_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'B2Co_combine_2C_vae')))

                BoC2_combine_2B_vae = torch.cat((B_z_1_vae, C_z_2_vae, B_z_3_vae, B_z_4_vae, B_z_5_vae), dim=1)
                BoC2_2B_vae = self.net.decoder(BoC2_combine_2B_vae)
                # pre_img_BoC2_combine_2B_vae = BoC2_combine_2B_vae.data.squeeze(0).cpu()
                # image_BoC2_combine_2B_vae = []
                # for item in range(len(pre_img_BoC2_combine_2B_vae)):
                #     image_BoC2_combine_2B_vae.append(unloader(F.sigmoid(pre_img_BoC2_combine_2B_vae[item].data)))
                #     image_BoC2_combine_2B_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'BoC2_combine_2B_vae')))

                # C D same font_color 3
                D3Co_combine_2C_vae = torch.cat((C_z_1_vae, C_z_2_vae, D_z_3_vae, C_z_4_vae, C_z_5_vae), dim=1)
                D3Co_2C_vae = self.net.decoder(D3Co_combine_2C_vae)
                # pre_img_D3Co_combine_2C_vae = D3Co_combine_2C_vae.data.squeeze(0).cpu()
                # image_D3Co_combine_2C_vae = []
                # for item in range(len(pre_img_D3Co_combine_2C_vae)):
                #     image_D3Co_combine_2C_vae.append(unloader(F.sigmoid(pre_img_D3Co_combine_2C_vae[item].data)))
                #     image_D3Co_combine_2C_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'D3Co_combine_2C_vae')))

                DoC3_combine_2D_vae = torch.cat((D_z_1_vae, D_z_2_vae, C_z_3_vae, D_z_4_vae, D_z_5_vae), dim=1)
                DoC3_2D_vae = self.net.decoder(DoC3_combine_2D_vae)
                # pre_img_DoC3_combine_2D_vae = DoC3_combine_2D_vae.data.squeeze(0).cpu()
                # image_DoC3_combine_2D_vae = []
                # for item in range(len(pre_img_DoC3_combine_2D_vae)):
                #     image_DoC3_combine_2D_vae.append(unloader(F.sigmoid(pre_img_DoC3_combine_2D_vae[item].data)))
                #     image_DoC3_combine_2D_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'DoC3_combine_2D_vae')))

                # C E same back_color 4
                E4Co_combine_2C_vae = torch.cat((C_z_1_vae, C_z_2_vae, C_z_3_vae, E_z_4_vae, C_z_5_vae), dim=1)
                E4Co_2C_vae = self.net.decoder(E4Co_combine_2C_vae)
                # pre_img_E4Co_combine_2C_vae = E4Co_combine_2C_vae.data.squeeze(0).cpu()
                # image_E4Co_combine_2C_vae = []
                # for item in range(len(pre_img_E4Co_combine_2C_vae)):
                #     image_E4Co_combine_2C_vae.append(unloader(F.sigmoid(pre_img_E4Co_combine_2C_vae[item].data)))
                #     image_E4Co_combine_2C_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'E4Co_combine_2C_vae')))

                EoC4_combine_2E_vae = torch.cat((E_z_1_vae, E_z_2_vae, E_z_3_vae, C_z_4_vae, E_z_5_vae), dim=1)
                EoC4_2E_vae = self.net.decoder(EoC4_combine_2E_vae)
                # pre_img_EoC4_combine_2E_vae = EoC4_combine_2E_vae.data.squeeze(0).cpu()
                # image_EoC4_combine_2E_vae = []
                # for item in range(len(pre_img_EoC4_combine_2E_vae)):
                #     image_EoC4_combine_2E_vae.append(unloader(F.sigmoid(pre_img_EoC4_combine_2E_vae[item].data)))
                #     image_EoC4_combine_2E_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'EoC4_combine_2E_vae')))

                # C F same style 5
                F5Co_combine_2C_vae = torch.cat((C_z_1_vae, C_z_2_vae, C_z_3_vae, C_z_4_vae, F_z_5_vae), dim=1)
                F5Co_2C_vae = self.net.decoder(F5Co_combine_2C_vae)
                # pre_img_F5Co_combine_2C_vae = F5Co_combine_2C_vae.data.squeeze(0).cpu()
                # image_F5Co_combine_2C_vae = []
                # for item in range(len(pre_img_F5Co_combine_2C_vae)):
                #     image_F5Co_combine_2C_vae.append(unloader(F.sigmoid(pre_img_F5Co_combine_2C_vae[item].data)))
                #     image_F5Co_combine_2C_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'F5Co_combine_2C_vae')))

                FoC5_combine_2F_vae = torch.cat((F_z_1_vae, F_z_2_vae, F_z_3_vae, F_z_4_vae, C_z_5_vae), dim=1)
                FoC5_2F_vae = self.net.decoder(FoC5_combine_2F_vae)
                # pre_img_FoC5_combine_2F_vae = FoC5_combine_2F_vae.data.squeeze(0).cpu()
                # image_FoC5_combine_2F_vae = []
                # for item in range(len(pre_img_FoC5_combine_2F_vae)):
                #     image_FoC5_combine_2F_vae.append(unloader(F.sigmoid(pre_img_FoC5_combine_2F_vae[item].data)))
                #     image_FoC5_combine_2F_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'FoC5_combine_2F_vae')))

                # combine_2C
                A1B2D3E4F5_combine_2C_vae = torch.cat((A_z_1_vae, B_z_2_vae, D_z_3_vae, E_z_4_vae, F_z_5_vae), dim=1)
                A1B2D3E4F5_2C_vae = self.net.decoder(A1B2D3E4F5_combine_2C_vae)
                # pre_img_A1B2D3E4F5_combine_2C_vae = A1B2D3E4F5_combine_2C_vae.data.squeeze(0).cpu()
                # image_A1B2D3E4F5_combine_2C_vae = []
                # for item in range(len(pre_img_A1B2D3E4F5_combine_2C_vae)):
                #     image_A1B2D3E4F5_combine_2C_vae.append(unloader(F.sigmoid(pre_img_A1B2D3E4F5_combine_2C_vae[item].data)))
                #     image_A1B2D3E4F5_combine_2C_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'A1B2D3E4F5_combine_2C_vae')))

                # '''  need unsupervise '''
                A2B3D4E5F1_combine_2N_vae = torch.cat((F_z_1_vae, A_z_2_vae, B_z_3_vae, D_z_4_vae, E_z_5_vae), dim=1)
                A2B3D4E5F1_2N_vae = self.net.decoder(A2B3D4E5F1_combine_2N_vae)
                # pre_img_A2B3D4E5F1_combine_2N_vae = A2B3D4E5F1_combine_2N_vae.data.squeeze(0).cpu()
                # image_A2B3D4E5F1_combine_2N_vae = []
                # for item in range(len(pre_img_A2B3D4E5F1_combine_2N_vae)):
                #     image_A2B3D4E5F1_combine_2N_vae.append(unloader(F.sigmoid(pre_img_A1Co_combine_2C_vae[item].data)))
                #     image_A2B3D4E5F1_combine_2N_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'A2B3D4E5F1_combine_2N_vae')))

                A1Co_combine_2C_vae = torch.cat((A_z_1_vae, C_z_2_vae, C_z_3_vae, C_z_4_vae, C_z_5_vae), dim=1)
                A1Co_2C_vae = self.net.decoder(A1Co_combine_2C_vae)
                # pre_img_A1Co_combine_2C_vae = A1Co_2C_vae.data.squeeze(0).cpu()
                # image_A1Co_combine_2C_vae = []
                # for item in range(len(pre_img_A1Co_combine_2C_vae)):
                #     image_A1Co_combine_2C_vae.append(unloader(F.sigmoid(pre_img_A1Co_combine_2C_vae[item].data)))
                #     image_A1Co_combine_2C_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'A1Co_combine_2C_vae')))

                AoC1_combine_2A_vae = torch.cat((C_z_1_vae, A_z_2_vae, A_z_3_vae, A_z_4_vae, A_z_5_vae), dim=1)
                AoC1_2A_vae = self.net.decoder(AoC1_combine_2A_vae)
                # pre_img_AoC1_combine_2A_vae = AoC1_combine_2A_vae.data.squeeze(0).cpu()
                # image_AoC1_combine_2A_vae = []
                # for item in range(len(pre_img_AoC1_combine_2A_vae)):
                #     image_AoC1_combine_2A_vae.append(unloader(F.sigmoid(pre_img_AoC1_combine_2A_vae[item].data)))
                #     image_AoC1_combine_2A_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'AoC1_combine_2A_vae')))

                # C B same size 2
                B2Co_combine_2C_vae = torch.cat((C_z_1_vae, B_z_2_vae, C_z_3_vae, C_z_4_vae, C_z_5_vae), dim=1)
                B2Co_2C_vae = self.net.decoder(B2Co_combine_2C_vae)
                # pre_img_B2Co_combine_2C_vae = B2Co_combine_2C_vae.data.squeeze(0).cpu()
                # image_B2Co_combine_2C_vae = []
                # for item in range(len(pre_img_B2Co_combine_2C_vae)):
                #     image_B2Co_combine_2C_vae.append(unloader(F.sigmoid(pre_img_B2Co_combine_2C_vae[item].data)))
                #     image_B2Co_combine_2C_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'B2Co_combine_2C_vae')))

                BoC2_combine_2B_vae = torch.cat((B_z_1_vae, C_z_2_vae, B_z_3_vae, B_z_4_vae, B_z_5_vae), dim=1)
                BoC2_2B_vae = self.net.decoder(BoC2_combine_2B_vae)
                # pre_img_BoC2_combine_2B_vae = BoC2_combine_2B_vae.data.squeeze(0).cpu()
                # image_BoC2_combine_2B_vae = []
                # for item in range(len(pre_img_BoC2_combine_2B_vae)):
                #     image_BoC2_combine_2B_vae.append(unloader(F.sigmoid(pre_img_BoC2_combine_2B_vae[item].data)))
                #     image_BoC2_combine_2B_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'BoC2_combine_2B_vae')))

                # C D same font_color 3
                D3Co_combine_2C_vae = torch.cat((C_z_1_vae, C_z_2_vae, D_z_3_vae, C_z_4_vae, C_z_5_vae), dim=1)
                D3Co_2C_vae = self.net.decoder(D3Co_combine_2C_vae)
                # pre_img_D3Co_combine_2C_vae = D3Co_combine_2C_vae.data.squeeze(0).cpu()
                # image_D3Co_combine_2C_vae = []
                # for item in range(len(pre_img_D3Co_combine_2C_vae)):
                #     image_D3Co_combine_2C_vae.append(unloader(F.sigmoid(pre_img_D3Co_combine_2C_vae[item].data)))
                #     image_D3Co_combine_2C_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'D3Co_combine_2C_vae')))

                DoC3_combine_2D_vae = torch.cat((D_z_1_vae, D_z_2_vae, C_z_3_vae, D_z_4_vae, D_z_5_vae), dim=1)
                DoC3_2D_vae = self.net.decoder(DoC3_combine_2D_vae)
                # pre_img_DoC3_combine_2D_vae = DoC3_combine_2D_vae.data.squeeze(0).cpu()
                # image_DoC3_combine_2D_vae = []
                # for item in range(len(pre_img_DoC3_combine_2D_vae)):
                #     image_DoC3_combine_2D_vae.append(unloader(F.sigmoid(pre_img_DoC3_combine_2D_vae[item].data)))
                #     image_DoC3_combine_2D_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'DoC3_combine_2D_vae')))

                # C E same back_color 4
                E4Co_combine_2C_vae = torch.cat((C_z_1_vae, C_z_2_vae, C_z_3_vae, E_z_4_vae, C_z_5_vae), dim=1)
                E4Co_2C_vae = self.net.decoder(E4Co_combine_2C_vae)
                # pre_img_E4Co_combine_2C_vae = E4Co_combine_2C_vae.data.squeeze(0).cpu()
                # image_E4Co_combine_2C_vae = []
                # for item in range(len(pre_img_E4Co_combine_2C_vae)):
                #     image_E4Co_combine_2C_vae.append(unloader(F.sigmoid(pre_img_E4Co_combine_2C_vae[item].data)))
                #     image_E4Co_combine_2C_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'E4Co_combine_2C_vae')))

                EoC4_combine_2E_vae = torch.cat((E_z_1_vae, E_z_2_vae, E_z_3_vae, C_z_4_vae, E_z_5_vae), dim=1)
                EoC4_2E_vae = self.net.decoder(EoC4_combine_2E_vae)
                # pre_img_EoC4_combine_2E_vae = EoC4_combine_2E_vae.data.squeeze(0).cpu()
                # image_EoC4_combine_2E_vae = []
                # for item in range(len(pre_img_EoC4_combine_2E_vae)):
                #     image_EoC4_combine_2E_vae.append(unloader(F.sigmoid(pre_img_EoC4_combine_2E_vae[item].data)))
                #     image_EoC4_combine_2E_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'EoC4_combine_2E_vae')))

                # C F same style 5
                F5Co_combine_2C_vae = torch.cat((C_z_1_vae, C_z_2_vae, C_z_3_vae, C_z_4_vae, F_z_5_vae), dim=1)
                F5Co_2C_vae = self.net.decoder(F5Co_combine_2C_vae)
                # pre_img_F5Co_combine_2C_vae = F5Co_combine_2C_vae.data.squeeze(0).cpu()
                # image_F5Co_combine_2C_vae = []
                # for item in range(len(pre_img_F5Co_combine_2C_vae)):
                #     image_F5Co_combine_2C_vae.append(unloader(F.sigmoid(pre_img_F5Co_combine_2C_vae[item].data)))
                #     image_F5Co_combine_2C_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'F5Co_combine_2C_vae')))

                FoC5_combine_2F_vae = torch.cat((F_z_1_vae, F_z_2_vae, F_z_3_vae, F_z_4_vae, C_z_5_vae), dim=1)
                FoC5_2F_vae = self.net.decoder(FoC5_combine_2F_vae)
                # pre_img_FoC5_combine_2F_vae = FoC5_combine_2F_vae.data.squeeze(0).cpu()
                # image_FoC5_combine_2F_vae = []
                # for item in range(len(pre_img_FoC5_combine_2F_vae)):
                #     image_FoC5_combine_2F_vae.append(unloader(F.sigmoid(pre_img_FoC5_combine_2F_vae[item].data)))
                #     image_FoC5_combine_2F_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'FoC5_combine_2F_vae')))

                # combine_2C
                A1B2D3E4F5_combine_2C_vae = torch.cat((A_z_1_vae, B_z_2_vae, D_z_3_vae, E_z_4_vae, F_z_5_vae), dim=1)
                A1B2D3E4F5_2C_vae = self.net.decoder(A1B2D3E4F5_combine_2C_vae)
                # pre_img_A1B2D3E4F5_combine_2C_vae = A1B2D3E4F5_combine_2C_vae.data.squeeze(0).cpu()
                # image_A1B2D3E4F5_combine_2C_vae = []
                # for item in range(len(pre_img_A1B2D3E4F5_combine_2C_vae)):
                #     image_A1B2D3E4F5_combine_2C_vae.append(unloader(F.sigmoid(pre_img_A1B2D3E4F5_combine_2C_vae[item].data)))
                #     image_A1B2D3E4F5_combine_2C_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'A1B2D3E4F5_combine_2C_vae')))
                # '''  need unsupervise '''
                A2B3D4E5F1_combine_2N_vae = torch.cat((F_z_1_vae, A_z_2_vae, B_z_3_vae, D_z_4_vae, E_z_5_vae), dim=1)
                A2B3D4E5F1_2N_vae = self.net.decoder(A2B3D4E5F1_combine_2N_vae)
                # pre_img_A2B3D4E5F1_combine_2N_vae = A2B3D4E5F1_combine_2N_vae.data.squeeze(0).cpu()
                # image_A2B3D4E5F1_combine_2N_vae = []
                # for item in range(len(pre_img_A2B3D4E5F1_combine_2N_vae)):
                #     image_A2B3D4E5F1_combine_2N_vae.append(unloader(F.sigmoid(pre_img_A2B3D4E5F1_combine_2N_vae[item].data)))
                #     image_A2B3D4E5F1_combine_2N_vae[item].save(os.path.join(dir + '/sample_img',
                #                                                       '{}-{}-{}_img.png'.format(self.global_iter, item,
                #                                                                                 'A2B3D4E5F1_combine_2N_vae')))



                '''
                optimize for autoencoder
                '''

                # 1. recon_loss
                A_recon_loss = torch.mean(torch.abs(A_img - A_recon_vae))
                B_recon_loss = torch.mean(torch.abs(B_img - B_recon_vae))
                C_recon_loss = torch.mean(torch.abs(C_img - C_recon_vae))
                D_recon_loss = torch.mean(torch.abs(D_img - D_recon_vae))
                E_recon_loss = torch.mean(torch.abs(E_img - E_recon_vae))
                F_recon_loss = torch.mean(torch.abs(F_img - F_recon_vae))
                recon_loss = A_recon_loss + B_recon_loss + C_recon_loss + D_recon_loss + E_recon_loss + F_recon_loss

                # 2. sup_combine_loss
                A1Co_2C_loss = torch.mean(torch.abs(C_img - A1Co_2C_vae))
                AoC1_2A_loss = torch.mean(torch.abs(A_img - AoC1_2A_vae))
                B2Co_2C_loss = torch.mean(torch.abs(C_img - B2Co_2C_vae))
                BoC2_2B_loss = torch.mean(torch.abs(B_img - BoC2_2B_vae))
                D3Co_2C_loss = torch.mean(torch.abs(C_img - D3Co_2C_vae))
                DoC3_2D_loss = torch.mean(torch.abs(D_img - DoC3_2D_vae))
                E4Co_2C_loss = torch.mean(torch.abs(C_img - E4Co_2C_vae))
                EoC4_2E_loss = torch.mean(torch.abs(E_img - EoC4_2E_vae))
                F5Co_2C_loss = torch.mean(torch.abs(C_img - F5Co_2C_vae))
                FoC5_2F_loss = torch.mean(torch.abs(F_img - FoC5_2F_vae))
                A1B2D3E4F5_2C_loss = torch.mean(torch.abs(C_img - A1B2D3E4F5_2C_vae))
                combine_sup_loss = A1Co_2C_loss + AoC1_2A_loss + B2Co_2C_loss + BoC2_2B_loss + D3Co_2C_loss + DoC3_2D_loss + E4Co_2C_loss + EoC4_2E_loss + F5Co_2C_loss + FoC5_2F_loss + A1B2D3E4F5_2C_loss

                # 3. unsup_combine_loss
                # _, _, _,A2B3D4E5F1_z = self.net(A2B3D4E5F1_2N_vae)
                # combine_unsup_loss = torch.mean(
                #     torch.abs(F_z_1_vae - A2B3D4E5F1_z[:, 0:self.z_size_start_dim])) + torch.mean(
                #     torch.abs(A_z_2_vae - A2B3D4E5F1_z[:, self.z_size_start_dim: self.z_font_color_start_dim])) \
                #                      + torch.mean(
                #     torch.abs(B_z_3_vae - A2B3D4E5F1_z[:, self.z_font_color_start_dim: self.z_back_color_start_dim])) \
                #                      + torch.mean(
                #     torch.abs(D_z_4_vae - A2B3D4E5F1_z[:, self.z_back_color_start_dim: self.z_style_start_dim])) \
                #                      + torch.mean(torch.abs(E_z_5_vae - A2B3D4E5F1_z[:, self.z_style_start_dim:]))
                combine_unsup_loss =combine_sup_loss
                #
                #
                #
                #
                # if self.objective == 'H':
                #     beta_vae_loss = recon_loss + self.beta*total_kld
                # elif self.objective == 'B':
                #     C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
                #     beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()
                #
                # # self.optim.zero_grad()
                # # beta_vae_loss.backward()
                # # self.optim.step()
                #
                # if self.viz_on and self.global_iter%self.gather_step == 0:
                #     self.gather.insert(iter=self.global_iter,
                #                        mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                #                        recon_loss=recon_loss.data, total_kld=total_kld.data,
                #                        dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)
                #     f = open(self.log_dir + '/log.txt', 'a')
                #     f.writelines(
                #         ['\n', '[{}] recon_loss:{}  total_kld:{}  mean_kld:{}'.format(
                #             self.global_iter, recon_loss.data, total_kld.data, mean_kld.data)])
                #     f.close()
                #
                if self.global_iter%self.display_step == 0:
                #     pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                #         self.global_iter, recon_loss.data, total_kld.data[0], mean_kld.data[0]))
                #
                #     var = logvar.exp().mean(0).data
                #     var_str = ''
                #     for j, var_j in enumerate(var):
                #         var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                #     pbar.write(var_str)
                #
                #     if self.objective == 'B':
                #         pbar.write('C:{:.3f}'.format(C.data[0]))
                #
                    if self.viz_on:
                        self.gather.insert(images=A_img.data)
                        self.gather.insert(images=B_img.data)
                        self.gather.insert(images=C_img.data)
                        self.gather.insert(images=D_img.data)
                        self.gather.insert(images=E_img.data)
                        self.gather.insert(images=F_img.data)
                        self.gather.insert(images=F.sigmoid(A_recon_vae).data)
                        self.viz_reconstruction()
                        # self.viz_lines()
                        '''
                        combine show
                        '''
                        self.gather.insert(combine_supimages=F.sigmoid(AoC1_2A_vae).data)
                        self.gather.insert(combine_supimages=F.sigmoid(BoC2_2B_vae).data)
                        self.gather.insert(combine_supimages=F.sigmoid(D3Co_2C_vae).data)
                        self.gather.insert(combine_supimages=F.sigmoid(DoC3_2D_vae).data)
                        self.gather.insert(combine_supimages=F.sigmoid(EoC4_2E_vae).data)
                        self.gather.insert(combine_supimages=F.sigmoid(FoC5_2F_vae).data)
                        self.viz_combine_recon()

                        self.gather.insert(combine_unsupimages=F.sigmoid(A1B2D3E4F5_2C_vae).data)
                        self.gather.insert(combine_unsupimages=F.sigmoid(A2B3D4E5F1_2N_vae).data)
                        self.viz_combine_unsuprecon()
                        # self.viz_combine(x)
                        self.gather.flush()
                #
                #     if self.viz_on or self.save_output:
                #         self.viz_traverse()

                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter%10000 == 0:
                    self.save_checkpoint(str(self.global_iter))

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
        self.viz.images(images, env=self.viz_name + '_reconstruction',
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
        self.viz.images(images, env=self.viz_name + 'combine_supimages',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.save_sample_img(images, 'combine_sup')

    def viz_combine_unsuprecon(self):
        # self.net_mode(train=False)
        A1B2D3E4F5_2C = self.gather.data['combine_unsupimages'][0][:100]
        A1B2D3E4F5_2C = make_grid(A1B2D3E4F5_2C, normalize=True)
        A2B3D4E5F1_2N = self.gather.data['combine_unsupimages'][1][:100]
        A2B3D4E5F1_2N = make_grid(A2B3D4E5F1_2N, normalize=True)
        images = torch.stack([A1B2D3E4F5_2C, A2B3D4E5F1_2N], dim=0).cpu()
        self.viz.images(images, env=self.viz_name + 'combine_unsupimages',
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
        self.viz.images(images, env=self.viz_name + '_combine',
                        opts=dict(title=str(self.global_iter)), nrow=10)

    def viz_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()

        mus = torch.stack(self.gather.data['mu']).cpu()
        vars = torch.stack(self.gather.data['var']).cpu()

        dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        mean_klds = torch.stack(self.gather.data['mean_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1).cpu()
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

        if self.win_kld is None:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))
        else:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        win=self.win_kld,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))

        if self.win_mu is None:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=mus,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior mean',))
        else:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_mu,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior mean',))

        if self.win_var is None:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior variance',))
        else:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_var,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior variance',))
        self.net_mode(train=True)

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

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        Auto_path = os.path.join(self.model_save_dir, self.viz_name, '{}-Auto.ckpt'.format(resume_iters))
        self.Autoencoder.load_state_dict(torch.load(Auto_path, map_location=lambda storage, loc: storage))
        print("=> loaded checkpoint '{} (iter {})'".format(self.viz_name, resume_iters))
