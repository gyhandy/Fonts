"""main.py"""

import argparse

import numpy as np
import torch

from solver_supervised_fonts import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)

    if args.train:
        net.train()
    else:
        net.traverse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=3, type=int, help='batch size')
    # model params
    parser.add_argument('--crop_size', type=int, default=208, help='crop size for the ilab dataset')
    parser.add_argument('--image_size', type=int, default=128, help='crop size for the ilab dataset')
    parser.add_argument('--c_dim', type=int, default=6, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    # parser.add_argument('--g_repeat_num', type=int, default=2, help='number of residual blocks in G for encoder and decoder')
    parser.add_argument('--g_repeat_num', type=int, default=1,
                        help='number of residual blocks in G for encoder and decoder')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--d_pose_repeat_num', type=int, default=2, help='number of strided conv layers in D')
    parser.add_argument('--lambda_combine', type=float, default=1, help='weight for lambda_combine')
    parser.add_argument('--lambda_unsup', default=0, type=float, help='lambda_recon')
    parser.add_argument('--lambda_GAN', default=1, type=float, help='lambda_recon')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

    # parser.add_argument('--z_dim', default=1000, type=int, help='dimension of the representation z')
    parser.add_argument('--z_dim', default=100, type=int, help='dimension of the representation z')
    '''
    the weight for pose and background
    '''
    parser.add_argument('--z_content', default=20, type=int, help='dimension of the z_content in z')
    parser.add_argument('--z_size', default=20, type=int, help='dimension of the z_size in z')
    parser.add_argument('--z_font_color', default=20, type=int, help='dimension of the z_font_color in z')
    parser.add_argument('--z_back_color', default=20, type=int, help='dimension of the z_back_color in z')
    parser.add_argument('--z_style', default=20, type=int, help='dimension of the z_style in z')

    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--objective', default='H', type=str, help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='fonts_unsup_nswap', type=str, help='dataset name')
    # parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=32, type=int, help='dataloader num_workers')

    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='fonts_Nswap', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    '''
    save model
    '''
    # parser.add_argument('--model_save_dir', default='checkpoints', type=str, help='output directory')
    parser.add_argument('--model_save_dir', default='checkpoints', type=str, help='output directory')
    parser.add_argument('--resume_iters', type=int, default=0, help='resume training from this step')

    parser.add_argument('--gather_step', default=10000, type=int, help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=10000, type=int, help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=10000, type=int, help='number of iterations after which a checkpoint is saved')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--use_server', default='False', type=str2bool,
                        help='use server to train the model need change the data location')
    parser.add_argument('--which_server', default='15', type=str,
                        help='use which server to train the model 15 or 21')


    args = parser.parse_args()

    main(args)
