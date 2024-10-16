"""Generic training script that train a model using a given dataset."""

import argparse
import sys
from timeit import default_timer as timer

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import utils

parser = argparse.ArgumentParser("Evaluating a model")
parser.add_argument('--model_path', type=str,
                    default='examples/pre_trained_models/squeeze_complex_bypass.pt', help='model directory')
parser.add_argument('--dataset_path', type=str,
                    default='data', help='dataset directory')
parser.add_argument('--model_name', type=str,
                    default='SqueezeNet', help='name of model')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--trec', type=str, default=[0]*26,
                    action=utils.SplitArgs, help='indication of using trec on each conv layer')
parser.add_argument('--L', type=str, default=[1]*26, action=utils.SplitArgs,
                    help='L of each conv layer')
parser.add_argument('--H', type=str, default=[1]*26, action=utils.SplitArgs,
                    help='H of each conv layer')
parser.add_argument('--bp_trec', type=str, default=[0]*4,
                    action=utils.SplitArgs, help='indication of using trec on each conv layer')
parser.add_argument('--bp_L', type=str, default=[1]*4, action=utils.SplitArgs,
                    help='L of each conv layer')
parser.add_argument('--bp_H', type=str, default=[1]*4, action=utils.SplitArgs,
                    help='H of each conv layer')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='verbose mode')
args = parser.parse_args()


'''Load test data'''


def load_test_data(batch_size=100):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True,
                                             shuffle=False, num_workers=0)
    return testset, testloader


'''Evaluation'''


def test(net, testset, testloader):
    correct = 0
    total = 0
    start = timer()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            images = images.cuda()

            time_start = timer()

            outputs = net(images).cpu()

            if args.verbose:
                torch.cuda.synchronize()
                time_end = timer()
                print(f'Inference time (ms): {1000*(time_end - time_start)}')
                print(f'Sum. Inference time (ms): {1000*(time_end - start)}')
                print(f'Batch {i+1}/{len(testloader)}')

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    torch.cuda.synchronize()
    print('Inference time (sec): {}'.format(timer() - start))
    print('Accuracy of the network on the 10000 test images: {}'.format(correct / total))
    return correct / total


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(args.gpu)

    testset, testloader = load_test_data(args.batch_size)

    net = utils.get_network(args)
    net.eval()

    fallback_trad_conv = not any(args.trec)
    if fallback_trad_conv:
        net.load_state_dict(torch.load(f=args.model_path), strict=False)
    else:
        net.load_state_dict(torch.load(args.model_path))

    net = net.cuda()

    torch.backends.cudnn.enabled = False

    test(net, testset, testloader)


if __name__ == '__main__':
    main()
