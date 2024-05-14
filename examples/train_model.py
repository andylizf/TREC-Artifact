
"""Generic training script that train a model using a given dataset."""

import argparse
import logging
import os
import sys
import time
from timeit import default_timer as timer

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import utils

parser = argparse.ArgumentParser("Train from scratch.")
parser.add_argument('--checkpoint_path', type=str,
                    default='EXP', help='checkpoint and logging directory')
parser.add_argument('--dataset_path', type=str,
                    default='data', help='dataset directory')
parser.add_argument('--model_name', type=str,
                    default='SqueezeNet', help='name of model')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=0.0001, help='weight decay')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--step', type=float, default=15, help='scheduler step')
parser.add_argument('--trec', type=str, default=[0]*2, action=utils.SplitArgs,
                    help='indication of using trec on each conv layer')
parser.add_argument('--L', type=str, default=[1]*2, action=utils.SplitArgs,
                    help='L of each conv layer')
parser.add_argument('--H', type=str, default=[1]*2, action=utils.SplitArgs,
                    help='H of each conv layer')
parser.add_argument('--bp_trec', type=str, default=[0]*4,
                    action=utils.SplitArgs, help='indication of using trec on each conv layer')
parser.add_argument('--bp_L', type=str, default=[1]*4, action=utils.SplitArgs,
                    help='L of each conv layer')
parser.add_argument('--bp_H', type=str, default=[1]*4, action=utils.SplitArgs,
                    help='H of each conv layer')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
args = parser.parse_args()


'''Logging settings'''
utils.create_exp_dir(args.checkpoint_path)
args.checkpoint_path = '{}/{}-{}'.format(
    args.checkpoint_path, args.model_name, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.checkpoint_path, scripts_to_save=glob.glob('*.py'))
utils.create_exp_dir(args.checkpoint_path)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.checkpoint_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


'''Evaluation'''


def test(net, testset, testloader):
    correct = 0
    total = 0
    start = timer()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            images = images.cuda()
            outputs = net(images).cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logging.info('Inference time (sec): {}'.format(timer() - start))
    logging.info(
        'Accuracy of the network on the 10000 test images: {}'.format(correct / total))
    return correct / total


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    net = utils.get_network(args)
    net = net.cuda()

    max_acc = 0
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, args.step, gamma=0.1, last_epoch=-1)
    for epoch in range(args.epochs):
        running_loss = 0.0
        net.train()

        '''Train'''
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.cuda()).cpu()
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            l = loss.detach().item()
            running_loss += l
            optimizer.step()
            if i % 100 == 99:
                logging.info('[epoch=%d, batch=%5d] loss: %.3f',
                             epoch, i + 1, running_loss / 100)
                running_loss = 0.0
        scheduler.step()

        '''Evaluation'''
        net.eval()
        acc = test(net, testset, testloader)
        if acc > max_acc:
            max_acc = acc
            logging.info("Saving currently best model..")
            utils.save(net, os.path.join(args.checkpoint_path, 'weights.pt'))
        if epoch == 49:
            logging.info("Saving model in the 50th epoch..")
            utils.save(net, os.path.join(
                args.checkpoint_path, 'weights_50.pt'))
            test(net, testset, testloader)

    logging.info("Saving model..")
    utils.save(net, os.path.join(args.checkpoint_path, 'weights_100.pt'))
    test(net, testset, testloader)
    logging.info('Finished Training')


if __name__ == '__main__':
    main()
