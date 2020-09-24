import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from PIL import Image
import resnet
from data_loader import data_loader
from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate


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


def get_model(model_name, weight_path=None):
    if model_name == 'resnet18':
        model = resnet.resnet18(pretrained=args.pretrained)
    elif model_name == 'AlexNet':
        model = AlexNet(52)
    else:
        raise ValueError('invalid network name:{}'.format(model_name))

    if weight_path is not None:
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

    return model


def main(args):
    # create model
    model = get_model(args.model_name, args.biased_weight_path)
    checkpoint = torch.load(args.biased_weight_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()

    # Data loading
    loaders = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)
    validate(loaders, model, args.save_dir)


def validate(loaders, model, save_dir):
    exec('log_txt=' + open('/lab/tmpig23b/u/zhix/font_dataset_D/log.txt').read())
    dset_dic = {}
    for key, value in locals()['log_txt'].items():
        if len(value) == 1:
            dset_dic[key] = 'G1'
        elif len(value) == 3:
            dset_dic[key] = 'G2'
        else:
            dset_dic[key] = 'G3'

    for dset in ['G1', 'G2','G3']:
        dset_dir = os.path.join(save_dir, dset)
        if not os.path.exists(dset_dir):
            os.makedirs(dset_dir)


    for loader in loaders:
        for i, (img,label) in enumerate(tqdm(loader)):
            out = model(img.cuda())
            _, pre = torch.max(out.data, 1)
            if pre != label.cuda():
                image_dir, _ = loader.dataset.samples[i]
                image = Image.open(image_dir)
                image_name = os.path.basename(image_dir)
                image_name = os.path.splitext(image_name)[0]
                _, letter, bg, _1, _2, _3 = image_name.split('_')
                image.save(os.path.join(save_dir, dset_dic[letter], letter+'_'+bg+'_'+str(i) + '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='resnet18', help='ImageNet classification network')
    parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--data', default='/lab/tmpig23b/u/zhix/classifiction_dataset2', help='path to dataset')
    parser.add_argument('--log_dir', default='/lab/tmpig23b/u/zhix/font_dataset_D/log.txt')

    parser.add_argument('--biased-weight-path', type=str, default=
                        '/lab/tmpig23b/u/zhix/resnet18_bias2/29.pth', help='weight path of the biased model')
    parser.add_argument('--save_dir', default='/lab/tmpig23b/u/yao-data/fonts_Nswap/GradCam/wrong_pics/resnet18',
                        help='the path for wrong pictures')
    parser.add_argument('-b', '--batch-size', default=1, type=int, help='mini-batch size')
    parser.add_argument('-j', '--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                        help='use pin memory')

    args = parser.parse_args()
    main(args)