"""solver.py"""

import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif
import model
from load_purity_dataset import data_loader


class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.model_save_dir = args.model_save_dir
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
        '''arrangement for each domain'''
        self.nc = 3
        self.decoder_dist = 'gaussian'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model
        # self.Autoencoder = Generator(self.nc, self.g_conv_dim, self.g_repeat_num)
        self.Autoencoder = model.AE_H(z_dim=2)

        self.auto_optim = optim.Adam(list(self.Autoencoder.parameters()), lr=self.lr, betas=(self.beta1, self.beta2))
        # self.Autoencoder = BetaVAE_ilab(self.z_dim, self.nc)

        self.Autoencoder.to(self.device)

        self.batch_size = args.batch_size
        self.data_loader, self.val_loader = data_loader()

    def train(self):
        # self.net_mode(train=True)
        # Start training from scratch or resume training.
        self.global_iter = 0

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        for _ in range(10000):
            for _, (img, label) in enumerate(self.data_loader):
                # appe, pose, combine
                self.global_iter += 1
                pbar.update(1)

                img = Variable(cuda(img, self.use_cuda))

                ## 1. A B C seperate(first400: id last600 background)
                img_recon, img_z = self.Autoencoder(img)
                ''' refer 1: content, 2: size, 3: font-color, 4 back_color, 5 style'''

                img_recon_loss = torch.mean(torch.abs(img - img_recon))
                recon_loss = img_recon_loss

                vae_unsup_loss = recon_loss
                self.auto_optim.zero_grad()
                vae_unsup_loss.backward()
                self.auto_optim.step()
                log = open('log.txt', 'a')
                log.write("epoch" + str(self.global_iter) + ':' + str(recon_loss.item()))
                log.write('\n')
                img_z = img_z.to(torch.device('cpu'))
                log.write(str(label.tolist()) + ":" + str(img_z.tolist()))
                log.write('\n')
                log.close()
                if self.global_iter % 500 == 0:
                    for _, (img, label) in enumerate(self.val_loader):
                        img = Variable(cuda(img, self.use_cuda))
                        img_recon, img_z = self.Autoencoder(img)
                        img_recon = img_recon.to(torch.device('cpu'))
                        global_iter_save_dir = os.path.join(self.model_save_dir,
                                                str(self.global_iter) + '_' + "recon_" + str(label.item()) + '.png')
                        save_image(img_recon, global_iter_save_dir)

        pbar.write("[Training Finished]")
        pbar.close()

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(state, filename='checkpoint.pth'):
        torch.save(state, filename)