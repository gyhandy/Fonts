# latent classifier ?
# generated image classifier?
#
# imageClassifier
from torchvision.models import resnet18
import torch.optim as optim
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import model as alexnet
#from data_loader import data_loader
from utils import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate
from torchvision.models import resnet18

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data',default='/lab/tmpig23b/u/he-data/small_training_image_dataset.npy', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='/lab/tmpig23b/u/zhix/resnet18_bias2/',
                    help='model architecture')
parser.add_argument('--epochs', default=90, type=int,
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=0, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str,
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate',dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0.0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    model = resnet18(pretrained=args.pretrained)

    # use cuda
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading
    train_image_set = np.load('/lab/tmpig23b/u/he-data/fonts_dataset_small_540.npy')
    train_label     = np.load('/lab/tmpig23b/u/he-data/fonts_label_small_540.npy')
    test_image_set  = np.load('/lab/tmpig23b/u/he-data/fonts_dataset_test_5400.npy')
    test_label      = np.load('/lab/tmpig23b/u/he-data/fonts_test_label_5400.npy')
    '''
    if args.evaluate:
        validate(train_loader2,val_loader, model, criterion,59)
        return
    '''

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_image_set, model, criterion, optimizer, epoch, args.print_freq, val_loader)
        validate(test_image_set, val_loader, model, criterion, epoch)
        # evaluate on validation set
        # prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)

        # remember the best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, args.arch + str(epoch) + '.pth')


def train(train_set,train_label, model, criterion, optimizer, epoch, print_freq):
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()

    for i in range(train_set.shape[0]):
        # measure data loading time

        target = target.cuda()
        input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        prec1, _ = accuracy(output.data, target, topk=(1,5))
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1))

def validate(train_loader, val_loader, model, criterion, epoch):
    correct = [0] * 53
    total = [0] * 53
    acc = [0] * 53
    for (img,label) in val_loader:
        #index = label.item()
        img = img.cuda()
        label = label.cuda()
        out = model(img)
        _, pre = torch.max(out.data, 1)
        total[0] += label.size(0)
        correct[0] += (pre == label).sum().item()
        for i in range(52):
            tmp = (torch.ones(label.size())) *i
            tmp = tmp.cuda()
            tmp = tmp.long()
            total[i+1] += (tmp == label).sum().item()
            correct[i+1] += ((tmp == label)*(pre == label)).sum().item()
        #total[index+1] += label.size(0)
        #correct[index+1] += (pre == label).sum().item()
    for i in range(53):
        acc[i] = correct[i]/total[i]
    print('test accuracy: {}'.format(correct[0] / total[0]))
    print(str(total))
    print(str(correct))
    log = open(args.arch+'log.txt', 'a')
    log.write("epoch"+str(epoch)+"in test:\n")
    log.write(str(acc))
    log.write('\n')
    log.close()

    correct = [0] * 53
    total = [0] * 53
    acc = [0] * 53
    for (img,label) in train_loader:
        #index = label.item()
        img = img.cuda()
        label = label.cuda()
        out = model(img)
        _, pre = torch.max(out.data, 1)
        total[0] += label.size(0)
        correct[0] += (pre == label).sum().item()
        for i in range(52):
            tmp = (torch.ones(label.size())) *i
            tmp = tmp.cuda()
            tmp = tmp.long()
            total[i+1] += (tmp == label).sum().item()
            correct[i+1] += ((tmp == label)*(pre == label)).sum().item()
    for i in range(53):
        acc[i] = correct[i]/total[i]
    print('train accuracy: {}'.format(correct[0] / total[0]))
    log = open(args.arch+'log.txt', 'a')
    log.write("epoch"+str(epoch)+"in train:\n")
    log.write(str(acc))
    log.write('\n')
    log.close()

if __name__ == '__main__':
    main()