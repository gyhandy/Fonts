# -*- coding: utf-8 -*-
"""
Created on 2020/9/14
@author: Ava
"""
import argparse
import os
import cv2
import numpy as np
import torch
from skimage import io
from torch import nn
import resnet
from grad_cam import GradCAM, GradCamPlusPlus
from guided_back_propagation import GuidedBackPropagation
from utils import cuda
from torch.autograd import Variable
from torchvision import transforms as T
from tqdm import tqdm


class AlexNet(nn.Module):
    def __init__(self, num_classes=52):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # input shape is 224 x 224 x 3
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # shape is 55 x 55 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape is 27 x 27 x 64

            nn.Conv2d(64, 192, kernel_size=5, padding=2), # shape is 27 x 27 x 192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape is 13 x 13 x 192

            nn.Conv2d(192, 384, kernel_size=3, padding=1), # shape is 13 x 13 x 384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # shape is 6 x 6 x 256
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def get_tensor_img(img):
    # transform = []
    # transform.append(T.Resize(256))
    # transform.append(T.CenterCrop(224))
    # transform.append(T.ToTensor())
    # transform.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    # transform = T.Compose(transform)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform(img)
    tensor_img = Variable(cuda(tensor_img, False))
    tensor_img = torch.tensor(tensor_img.unsqueeze(0), requires_grad=True)

    return tensor_img


def get_model(model_name, weight_path=None):
    if model_name == 'resnet18':
        model = resnet.resnet18(pretrained=args.pretrained)
    elif model_name == 'AlexNet':
        model = AlexNet(52)
    else:
        raise ValueError('invalid model name:{}'.format(model_name))

    if weight_path is not None:
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

    return model


def get_last_conv_name(net):
    """
    get the name of last conv layer
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_image(image_dir):
    image = io.imread(image_dir)
    image = np.float32(cv2.resize(image, (224, 224))) / 255

    return image


def prepare_input(image):
    image = image.copy()

    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    generate CAM picture
    :param image: [H,W,C], original picture
    :param mask: [H,W], range [0, 1]
    :return: tuple(cam,heatmap)
    """
    # transfer mask to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # combine heatmap and original picture
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    standard picture
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    generate guided back propagation
    :param grad: tensor,[3,H,W]
    :return:
    """
    # standardization
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def generate_results(args, image_dir, model, flag, save_dir):
    # input
    # img = io.imread(args.image_path)
    # img = np.float32(cv2.resize(img, (224, 224))) / 255
    # inputs = prepare_input(img)
    img = prepare_image(image_dir)
    inputs = prepare_input(img)

    # output pictures
    image_dict = {}
    # Grad-CAM
    layer_name = get_last_conv_name(model) if args.layer_name is None else args.layer_name
    grad_cam = GradCAM(model, layer_name)
    mask = grad_cam(inputs, args.class_id)  # cam mask
    image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
    grad_cam.remove_handlers()

    # Grad-CAM++
    grad_cam_plus_plus = GradCamPlusPlus(model, layer_name)
    mask_plus_plus = grad_cam_plus_plus(inputs, args.class_id)  # cam mask
    image_dict['cam++'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # GuidedBackPropagation
    gbp = GuidedBackPropagation(model)
    inputs.grad.zero_()
    grad = gbp(inputs)
    gb = gen_gb(grad)
    image_dict['gb'] = norm_image(gb)
    # generate Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]
    image_dict['cam_gb'] = norm_image(cam_gb)

    save_image(image_dict, save_dir, flag)


def save_image(image_dicts, save_dir, flag):
    for key, image in image_dicts.items():
        io.imsave(os.path.join(save_dir, '{}-{}.jpg'.format(flag, key)), image, check_contrast=False)


def walkdir(folder):
    """Walk through every files in a directory"""
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield os.path.abspath(os.path.join(dirpath, filename))



def main(args):
    # model
    # net = get_net(args.network, args.weight_path)
    biased_model = get_model(args.model_name, args.biased_weight_path)
    unbiased_model = get_model(args.model_name, args.unbiased_weight_path)


    for dset in ['G1', 'G2', 'G3']:
        dset_dir = os.path.join(args.data_dir, dset)
        # Precomputing files count
        filescount = 0
        for _ in tqdm(walkdir(dset_dir)):
            filescount += 1
        for image_dir in tqdm(walkdir(dset_dir), total=filescount):
            file_name = os.path.basename(image_dir)
            image_name = os.path.splitext(file_name)[0]
            letter, bg, _ = image_name.split('_')
            save_dir = os.path.join(args.output_dir, args.model_name, dset, letter, bg)
            if os.path.exists(save_dir):
                continue
            else:
                os.makedirs(save_dir)
                generate_results(args, image_dir, biased_model, 'biased', save_dir)
                generate_results(args, image_dir, unbiased_model, 'unbiased', save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='resnet18', help='ImageNet classification network')
    parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--data_dir', default= '/lab/tmpig23b/u/yao-data/fonts_Nswap/GradCam/wrong_pics/resnet18',
                        help='path to image')
    # parser.add_argument('--weight-path', type=str, default=
    #                     '/lab/tmpig23b/u/yao-data/directly_supervise/pretrain_swap_letter/ckpt/1100000-Image.ckpt',
    #                     help='weight path of the model')
    parser.add_argument('--biased-weight-path', type=str, default=
                        '/lab/tmpig23b/u/zhix/resnet18_bias2/29.pth', help='weight path of the biased model')
    parser.add_argument('--unbiased-weight-path', type=str, default=
                        '/lab/tmpig23b/u/zhix/resnet18_unbias2/22.pth', help='weight path of the unbiased model')

    parser.add_argument('--layer-name', type=str, default=None, help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None, help='class id')
    parser.add_argument('--output-dir', type=str, default='/lab/tmpig23b/u/yao-data/fonts_Nswap/GradCam',
                        help='output directory to save results')

    args = parser.parse_args()

    main(args)