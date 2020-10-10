# python3.7
"""Trains semantic boundary from latent space.

Basically, this file takes a collection of `latent code - attribute score`
pairs, and find the separation boundary by treating it as a bi-classification
problem and training a linear SVM classifier. The well-trained decision boundary
of the SVM classifier will be saved as the boundary corresponding to a
particular semantic from the latent space. The normal direction of the boundary
can be used to manipulate the correpsonding attribute of the synthesis.
"""

import os.path
import argparse
from sklearn import svm
from data_loader import data_loader
from logger import setup_logger
from utils import str2bool
import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from utils import cuda, grid2gif
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
# import sklearn.external.joblib as extjoblib
import joblib



class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator_fc(nn.Module):
    """Generator network, with fully connected layers"""
    def __init__(self, nc=3, conv_dim=64, repeat_num=2, z_dim=500):
        self.z_dim = z_dim
        super(Generator_fc, self).__init__()
        '''
        encoder
        '''
        self.start_layers = []
        # self.start_layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.start_layers.append(nn.Conv2d(nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.start_layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        self.start_layers.append(nn.ReLU(inplace=True))
        self.start_part = nn.Sequential(*self.start_layers)

        # Down-sampling layers.
        self.down_layers = []
        curr_dim = conv_dim
        for i in range(4):
            if i <= 1:
                self.down_layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
                self.down_layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
                self.down_layers.append(nn.ReLU(inplace=True))
                curr_dim = curr_dim * 2
            else:
                self.down_layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
                self.down_layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
                self.down_layers.append(nn.ReLU(inplace=True))

        self.down_part = nn.Sequential(*self.down_layers)
        self.eli_pose_part = nn.Sequential(*self.start_layers, *self.down_layers)

        # Bottleneck layers.
        self.bottle_encoder_layers = []
        for i in range(repeat_num):
            self.bottle_encoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.bottlen_encoder_part = nn.Sequential(*self.bottle_encoder_layers)

        self.encoder = nn.Sequential(*self.start_layers, *self.down_layers, *self.bottle_encoder_layers)
        # fc layers
        self.fc_encoder = nn.Sequential(
            nn.Linear(256 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Linear(4096, self.z_dim)
        )
        '''
        decoder
        '''
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 256 * 8 * 8)
        )
        self.bottle_decoder_layers = []
        for i in range(repeat_num):
            self.bottle_decoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.bottlen_decoder_part = nn.Sequential(*self.bottle_decoder_layers)

        # Up-sampling layers.
        self.up_layers = []
        for i in range(4):
            if i <= 1:
                self.up_layers.append(nn.ConvTranspose2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
                self.up_layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
                self.up_layers.append(nn.ReLU(inplace=True))
            else:
                self.up_layers.append(
                    nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
                self.up_layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
                self.up_layers.append(nn.ReLU(inplace=True))
                curr_dim = curr_dim // 2


        self.up_layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        self.up_layers.append(nn.Tanh())
        self.up_part = nn.Sequential(*self.up_layers)

        self.decoder = nn.Sequential(*self.bottle_decoder_layers, *self.up_layers)

        self.main = nn.Sequential(*self.start_layers, *self.down_layers, *self.bottle_encoder_layers, *self.bottle_decoder_layers, *self.up_layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or
        x1 = self.encoder(x)
        x1 = x1.view(x.shape[0], -1)
        z= self.fc_encoder(x1)
        x2 = self.fc_decoder(z)
        x2 = x2.view(x.shape[0], 256, 8, 8)
        x3 = self.decoder(x2)

        return x3, z



    def forward_origin(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=3):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    """
    Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


class Train_Boundary():
    def __init__(self, args, logger=None):
        self.logger = logger
        self.output_dir = args.output_dir
        self.project_dir = os.path.join(args.output_dir, args.project_name)
        if not os.path.exists(self.project_dir):
            os.makedirs(self.project_dir)

        self.load_models(args)
        self.prepare_data(args)


    def load_models(self, args):
        # model params
        self.image_size = args.image_size
        self.use_cuda = args.cuda and torch.cuda.is_available()
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
        self.Autoencoder = Generator_fc(3, args.g_conv_dim, args.g_repeat_num, self.z_dim)
        self.Autoencoder.to(self.device)
        self.Autoencoder.load_state_dict(torch.load(args.autoencoder_path, map_location=lambda storage, loc: storage))


        self.bg_classfier = resnet18()
        self.bg_classfier.cuda()
        self.bg_classfier.load_state_dict(torch.load(args.classifier_dir)['state_dict'])


    def prepare_data(self, args):
        self.encode_loader, self.classify_loader = data_loader(args)

        if not args.load_latent:
            self.logger.info('Begin compute latent codes.')
            latent_codes = self.compute_latent(args)
        else:
            latent_path = os.path.join(self.output_dir, 'latent_codes.npy')
            latent_codes = np.load(latent_path)

        if not args.load_scores:
            self.logger.info('Begin compute scores.')
            scores = self.compute_scores(args)
        else:
            scores_path = os.path.join(self.project_dir, 'scores.npy')
            scores = np.load(scores_path)

        self.logger.info('Begin Training.')
        self.tarin(latent_codes, scores)


    def encode_image(self, img):
        img = Variable(cuda(img, self.use_cuda))
        recon, z = self.Autoencoder(img)
        z_1 = z[:, 0:self.z_size_start_dim]  # 0-200
        z_2 = z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
        z_3 = z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
        z_4 = z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
        z_5 = z[:, self.z_style_start_dim:]  # 80-100
        return img, recon, z_1, z_2, z_3, z_4, z_5


    def compute_scores(self, args):
        scores = np.empty([0, 1])

        for (img, label) in tqdm(self.classify_loader):
            img = img.cuda()
            out = self.bg_classfier(img)
            tmp_scores = F.softmax(out, dim=1).data[:, args.attribute]
            scores = np.append(scores, torch.reshape(tmp_scores, (-1, 1)).cpu().numpy(), axis=0)
            np.save(os.path.join(self.project_dir, 'scores.npy'), scores)

        return scores


    def compute_latent(self, args):
        latent_codes = np.empty([0, args.z_back_color])

        for (img, label) in tqdm(self.encode_loader):
            img, recon, z_1, z_2, z_3, z_4, z_5 = self.encode_image(img)
            latent_codes = np.append(latent_codes, z_4.cpu().detach().numpy(), axis=0)
            np.save(os.path.join(self.output_dir, 'latent_codes.npy'), latent_codes)

        return latent_codes


    def tarin(self, latent_codes,
                       scores,
                       chosen_num_or_ratio=0.02,
                       split_ratio=0.7,
                       invalid_value=None,
                       ):
        """Trains boundary in latent space with offline predicted attribute scores.

        Given a collection of latent codes and the attribute scores predicted from the
        corresponding images, this function will train a linear SVM by treating it as
        a bi-classification problem. Basically, the samples with highest attribute
        scores are treated as positive samples, while those with lowest scores as
        negative. For now, the latent code can ONLY be with 1 dimension.

        NOTE: The returned boundary is with shape (1, latent_space_dim), and also
        normalized with unit norm.

        Args:
          latent_codes: Input latent codes as training data.
          scores: Input attribute scores used to generate training labels.
          chosen_num_or_ratio: How many samples will be chosen as positive (negative)
            samples. If this field lies in range (0, 0.5], `chosen_num_or_ratio *
            latent_codes_num` will be used. Otherwise, `min(chosen_num_or_ratio,
            0.5 * latent_codes_num)` will be used. (default: 0.02)
          split_ratio: Ratio to split training and validation sets. (default: 0.7)
          invalid_value: This field is used to filter out data. (default: None)
          logger: Logger for recording log messages. If set as `None`, a default
            logger, which prints messages from all levels to screen, will be created.
            (default: None)

        Returns:
          A decision boundary with type `numpy.ndarray`.

        Raises:
          ValueError: If the input `latent_codes` or `scores` are with invalid format.
        """
        if (not isinstance(latent_codes, np.ndarray) or
                not len(latent_codes.shape) == 2):
            raise ValueError(f'Input `latent_codes` should be with type'
                             f'`numpy.ndarray`, and shape [num_samples, '
                             f'latent_space_dim]!')
        num_samples = latent_codes.shape[0]
        latent_space_dim = latent_codes.shape[1]
        if (not isinstance(scores, np.ndarray) or not len(scores.shape) == 2 or
                not scores.shape[0] == num_samples or not scores.shape[1] == 1):
            raise ValueError(f'Input `scores` should be with type `numpy.ndarray`, and '
                             f'shape [num_samples, 1], where `num_samples` should be '
                             f'exactly same as that of input `latent_codes`!')
        if chosen_num_or_ratio <= 0:
            raise ValueError(f'Input `chosen_num_or_ratio` should be positive, '
                             f'but {chosen_num_or_ratio} received!')

        self.logger.info(f'Filtering training data.')
        if invalid_value is not None:
            latent_codes = latent_codes[scores[:, 0] != invalid_value]
            scores = scores[scores[:, 0] != invalid_value]

        self.logger.info(f'Sorting scores to get positive and negative samples.')
        sorted_idx = np.argsort(scores, axis=0)[::-1, 0]
        latent_codes = latent_codes[sorted_idx]
        scores = scores[sorted_idx]
        num_samples = latent_codes.shape[0]
        if 0 < chosen_num_or_ratio <= 1:
            chosen_num = int(num_samples * chosen_num_or_ratio)
        else:
            chosen_num = int(chosen_num_or_ratio)
        chosen_num = min(chosen_num, num_samples // 2)

        self.logger.info(f'Spliting training and validation sets:')
        train_num = int(chosen_num * split_ratio)
        val_num = chosen_num - train_num
        # Positive samples.
        positive_idx = np.arange(chosen_num)
        np.random.shuffle(positive_idx)
        positive_train = latent_codes[:chosen_num][positive_idx[:train_num]]
        positive_val = latent_codes[:chosen_num][positive_idx[train_num:]]
        attribute_mean = (np.mean(positive_val, axis=0) + np.mean(positive_train, axis=0)) / 2
        np.save(os.path.join(self.project_dir, 'attribute_mean.npy'), attribute_mean)
        # Negative samples.
        negative_idx = np.arange(chosen_num)
        np.random.shuffle(negative_idx)
        negative_train = latent_codes[-chosen_num:][negative_idx[:train_num]]
        negative_val = latent_codes[-chosen_num:][negative_idx[train_num:]]
        # Training set.
        train_data = np.concatenate([positive_train, negative_train], axis=0)
        train_label = np.concatenate([np.ones(train_num, dtype=np.int),
                                      np.zeros(train_num, dtype=np.int)], axis=0)
        self.logger.info(f'  Training: {train_num} positive, {train_num} negative.')
        # Validation set.
        val_data = np.concatenate([positive_val, negative_val], axis=0)
        val_label = np.concatenate([np.ones(val_num, dtype=np.int),
                                    np.zeros(val_num, dtype=np.int)], axis=0)
        self.logger.info(f'  Validation: {val_num} positive, {val_num} negative.')
        # Remaining set.
        remaining_num = num_samples - chosen_num * 2
        remaining_data = latent_codes[chosen_num:-chosen_num]
        remaining_scores = scores[chosen_num:-chosen_num]
        decision_value = (scores[0] + scores[-1]) / 2
        remaining_label = np.ones(remaining_num, dtype=np.int)
        remaining_label[remaining_scores.ravel() < decision_value] = 0
        remaining_positive_num = np.sum(remaining_label == 1)
        remaining_negative_num = np.sum(remaining_label == 0)
        self.logger.info(f'  Remaining: {remaining_positive_num} positive, '
                    f'{remaining_negative_num} negative.')

        self.logger.info(f'Training boundary.')
        clf = svm.SVC(kernel='linear')
        classifier = clf.fit(train_data, train_label)
        joblib.dump(classifier, os.path.join(self.project_dir, "svm_model.m"))
        self.logger.info(f'Finish training.')

        if val_num:
            val_prediction = classifier.predict(val_data)
            correct_num = np.sum(val_label == val_prediction)
            self.logger.info(f'Accuracy for validation set: '
                        f'{correct_num} / {val_num * 2} = '
                        f'{correct_num / (val_num * 2):.6f}')

        if remaining_num:
            remaining_prediction = classifier.predict(remaining_data)
            correct_num = np.sum(remaining_label == remaining_prediction)
            self.logger.info(f'Accuracy for remaining set: '
                        f'{correct_num} / {remaining_num} = '
                        f'{correct_num / remaining_num:.6f}')

        a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
        boundary = a / np.linalg.norm(a)
        np.save(os.path.join(self.project_dir, 'boundary.npy'), boundary)


def main(args):
  """Main function."""
  project_dir = os.path.join(args.output_dir, args.project_name)
  logger = setup_logger(project_dir, logger_name='generate_data')
  Train_Boundary(args, logger)


if __name__ == '__main__':
  parser = argparse.ArgumentParser( description='Train semantic boundary with given latent codes and attribute scores.')

  parser.add_argument('--attribute', default=2, type=int,
                      help='which attribute we want to find boundary(blue:0, green:1, red:2)')
  parser.add_argument('--project_name', default='center_classRGB_red', type=str, help='name to distinguish different boundary')

  parser.add_argument('--load_scores', default=True, type=str2bool, help='whether load score.npy')
  parser.add_argument('--load_latent', default=True, type=str2bool, help='whether load latent_codes.npy')

  parser.add_argument('-o', '--output_dir', default='/lab/tmpig23b/u/yao-data/generativeAE/find_boundary/back_color',
                      type=str, help='Directory to save the output results')
  parser.add_argument('-c', '--load_images_dir', default='/lab/tmpig23b/u/yao-data/generativeAE/dataset',
                      type=str, help='Path to the input images (required)')
  parser.add_argument('-b', '--batch-size', default=30, type=int, help='mini-batch size')
  parser.add_argument('-j', '--num_workers', default=0, type=int, help='number of data loading workers (default: 4)')
  parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true', help='use pin memory')

  parser.add_argument('--classifier_dir', default='/lab/tmpig23b/u/yao-data/generativeAE/model/resnet18/back_color/RGB/resnet18_1.pth',
                      type=str, help='Path to the back color classifier model.')

  parser.add_argument('--autoencoder_path', default='/lab/tmpig23b/u/zhix/interpolation/checkpoints/fonts_Nswap/970000-Auto.ckpt',
                      type=str, help='Path to the autoencoder model.')
  parser.add_argument('-n', '--chosen_num_or_ratio', type=float, default=0.02,
                      help='How many samples to choose for training.  (default: 0.02)')
  parser.add_argument('-r', '--split_ratio', type=float, default=0.7,
                      help='Ratio with which to split training and validation sets. (default: 0.7)')
  parser.add_argument('-V', '--invalid_value', type=float, default=None,
                      help='Sample whose attribute score is equal to this '
                           'field will be ignored. (default: None)')

  parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
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

  args = parser.parse_args()
  main(args)
