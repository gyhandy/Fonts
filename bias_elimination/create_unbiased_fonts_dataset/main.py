import argparse
import warnings
warnings.filterwarnings("ignore")
import os
from torchvision import transforms as T
import torch
from torch.autograd import Variable
from utils import cuda
from model_share import Generator_fc
from PIL import Image
from utils import str2bool
from torchvision import transforms


class Generate_pics(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.path = args.biased_dset_dir
        self.output_dir = args.output_dir
        self.unloader = transforms.ToPILImage()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder_dist = 'gaussian'
        self.dir_order = ['dset', 'character', 'bg', 'size', 'fg', 'style']

        transform = []
        # transform.append(T.Resize(128))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = T.Compose(transform)

        self.z_content_start_dim = 0
        self.z_size_start_dim = 20
        self.z_font_color_start_dim = 40
        self.z_back_color_start_dim = 60
        self.z_style_start_dim = 80

        self.main(args)


    def find_all_bg(self):
        self.letter_dict = {}
        self.bg_dict = {}

        letter_list = os.listdir(self.path)
        bg_list = []
        for letter in letter_list:
            dir = os.path.join(self.path, letter)
            self.letter_dict[letter] = os.listdir(dir)
            for bg in os.listdir(dir):
                self.bg_dict.setdefault(bg, []).append(letter)
            bg_list.extend(os.listdir(dir))
        self.bg_all_set = set(bg_list)


    def load_model(self, args):
        self.Autoencoder = Generator_fc(3, args.g_conv_dim, args.g_repeat_num, args.z_dim)
        self.Autoencoder.to(self.device)
        Auto_path = os.path.join(args.model_save_dir, '{}-Auto.ckpt'.format(args.resume_iters))
        self.Autoencoder.load_state_dict(torch.load(Auto_path, map_location=lambda storage, loc: storage))


    def process_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        tensor_img = Variable(cuda(image, self.use_cuda))

        recon, z = self.Autoencoder(tensor_img.unsqueeze(0))
        z_1 = z[:, 0:self.z_size_start_dim]  # 0-200
        z_2 = z[:, self.z_size_start_dim: self.z_font_color_start_dim]  # 20-40
        z_3 = z[:, self.z_font_color_start_dim: self.z_back_color_start_dim]  # 40-60
        z_4 = z[:, self.z_back_color_start_dim: self.z_style_start_dim]  # 60-80
        z_5 = z[:, self.z_style_start_dim:]  # 80-100

        return tensor_img, z_1, z_2, z_3, z_4, z_5


    def load_image(self, tensor_img):
        image = self.unloader(tensor_img.data[:100].cpu().clone().squeeze(0))
        return image


    def save_images(self, dic_images):
        for img_name, image in dic_images.items():
            dset, character, bg, size, fg, style = img_name.replace('.png', '').split('_')
            img_dir = os.path.join(self.output_dir, dset, character, bg, size, fg, style)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            image.save(os.path.join(img_dir, img_name))



    def main(self, args):
        self.find_all_bg()
        self.load_model(args)

        for letter in self.letter_dict:
            path = os.path.join(self.path, letter)
            for (dirpath, dirnames, filenames) in os.walk(path):
                for filename in filenames:
                    dic_images = {}
                    tensor_img, z_1, z_2, z_3, z_4, z_5 = self.process_image(os.path.join(dirpath, filename))
                    dic_images[filename] = self.load_image(tensor_img)
                    exec("dset, character, _, size, fg, style = filename.replace('.png', '').split('_')")
                    exist_bgs = self.letter_dict[letter]
                    unexist_bgs = list(self.bg_all_set - set(exist_bgs))

                    for unexist_bg in unexist_bgs:
                        swap_letter = self.bg_dict[unexist_bg][0]
                        for (tmp_dirpath, _, tmp_filenames) in os.walk(os.path.join(self.path, swap_letter)):
                            if tmp_filenames:
                                tmp_filename = tmp_filenames[0]
                                break
                        swap_img_path = os.path.join(tmp_dirpath, tmp_filename)
                        exec("_1, _2, bg, _3, _4, _5 = tmp_filename.replace('.png', '').split('_')")
                        _, swap_z_1, swap_z_2, swap_z_3, swap_z_4, swap_z_5 = self.process_image(swap_img_path)

                        swap_bg = torch.cat((z_1, z_2, z_3, swap_z_4, z_5), dim=1)
                        mid_swap_bg = self.Autoencoder.fc_decoder(swap_bg)
                        mid_swap_bg = mid_swap_bg.view(swap_bg.shape[0], 256, 8, 8)
                        swap_bg = self.Autoencoder.decoder(mid_swap_bg)
                        exec("image_name = '_'.join([dset, character, bg, size, fg, style])+'.png'")
                        dic_images[locals()['image_name']] = self.load_image(swap_bg)

                    self.save_images(dic_images)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    the weight for id and background
    '''
    parser.add_argument('--z_dim', default=100, type=int, help='dimension of the representation z')
    parser.add_argument('--z_content', default=20, type=int, help='dimension of the z_content in z')
    parser.add_argument('--z_size', default=20, type=int, help='dimension of the z_size in z')
    parser.add_argument('--z_font_color', default=20, type=int, help='dimension of the z_font_color in z')
    parser.add_argument('--z_back_color', default=20, type=int, help='dimension of the z_back_color in z')
    parser.add_argument('--z_style', default=20, type=int, help='dimension of the z_style in z')

    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--g_repeat_num', type=int, default=1,
                        help='number of residual blocks in G for encoder and decoder')
    '''
    load model
    '''
    parser.add_argument('--model_save_dir', default='/lab/tmpig23b/u/yao-data/fonts_Nswap/model/version2/biased',
                        type=str, help='model directory')
    parser.add_argument('--resume_iters', type=int, default=860000, help='resume training from this step')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    '''
    dataset
    '''
    parser.add_argument('--biased_dset_dir', default='/lab/tmpig23b/u/zhix/font_dataset_D/G1', type=str, help='biased dataset directory')
    parser.add_argument('--output_dir', default='/lab/tmpig23b/u/yao-data/fonts_Nswap/unbiased_dataset', type=str, help='output directory')

    args = parser.parse_args()
    Generate_pics(args)