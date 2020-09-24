import torch
import torchvision.transforms as T
from utils import cuda
from torch.autograd import Variable
from PIL import Image



red = '/lab/tmpig23b/u/yao-data/generationAE/dataset/purity/red/red.png'
blue = '/lab/tmpig23b/u/yao-data/generationAE/dataset/purity/blue/blue.png'
green = '/lab/tmpig23b/u/yao-data/generationAE/dataset/purity/green/green.png'


red_img = Image.open(red).convert('RGB')
blue_img = Image.open(blue).convert('RGB')
green_img = Image.open(green).convert('RGB')

transform = T.Compose([
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


red_img = transform(red_img)
red_img = Variable(cuda(red_img, True))
# tensor_img = torch.tensor(tensor_img.unsqueeze(0), requires_grad=True)

blue_img = transform(blue_img)
blue_img = Variable(cuda(blue_img, True))

green_img = transform(green_img)
green_img = Variable(cuda(green_img, True))


print(torch.mean(torch.abs(red_img - blue_img)))
print(torch.mean(torch.abs(red_img - green_img)))
print(torch.mean(torch.abs(blue_img - green_img)))