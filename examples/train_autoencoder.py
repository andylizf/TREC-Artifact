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
import torchvision.utils as vutils
import utils
import torch.nn.functional as F

parser = argparse.ArgumentParser("Train Autoencoder")
parser.add_argument('--checkpoint_path', type=str,
                    default='EXP', help='checkpoint and logging directory')
parser.add_argument('--dataset_path', type=str,
                    default='data', help='dataset directory')
parser.add_argument('--epochs', type=int, default=100,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=1e-4, help='init learning rate')
parser.add_argument('--weight_decay', type=float,
                    default=1e-6, help='weight decay')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--trec', type=str, default=[0]*16, action=utils.SplitArgs,
                    help='indication of using trec on each conv layer')
parser.add_argument('--L', type=str, default=[1]*16, action=utils.SplitArgs,
                    help='L of each conv layer')
parser.add_argument('--H', type=str, default=[1]*16, action=utils.SplitArgs,
                    help='H of each conv layer')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--model_name', type=str,
                    default='autoencoder_trec', help='name of model')

def test(net, testloader):
    net.eval()
    total_loss = 0
    total_psnr = 0  # Peak Signal-to-Noise Ratio
    total_ssim = 0  # Structural Similarity Index
    n = 0
    
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.cuda()
            decoded = net(inputs)
            
            # Calculate MSE loss
            loss = F.mse_loss(decoded, inputs)
            total_loss += loss.item()
            
            # Calculate PSNR
            mse = F.mse_loss(decoded, inputs, reduction='none').mean((1, 2, 3))
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            total_psnr += psnr.mean().item()
            
            # Calculate SSIM (structural similarity)
            ssim_val = ssim(decoded, inputs)
            total_ssim += ssim_val.item()
            
            n += 1
    
    avg_loss = total_loss / n
    avg_psnr = total_psnr / n
    avg_ssim = total_ssim / n
    
    metrics = {
        'loss': avg_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
    }
    
    logging.info(f'Test set: SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f}dB, Loss: {avg_loss:.4f}')
    return metrics

def ssim(img1, img2, window_size=11):
    """Calculate SSIM between two images"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def main():
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # Setup logging
    utils.create_exp_dir(args.checkpoint_path)
    args.checkpoint_path = '{}/autoencoder-{}'.format(
        args.checkpoint_path, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.checkpoint_path)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.checkpoint_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # GPU setup
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)
    
    # Create model and optimizer
    net = utils.get_network(args)
    net = net.cuda()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate,
                          weight_decay=args.weight_decay)
    
    best_metrics = {'psnr': 0, 'ssim': 0}
    
    # Training loop
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        time_start = timer()
        
        for i, (inputs, _) in enumerate(trainloader):
            inputs = inputs.cuda()
            
            # Forward pass
            decoded = net(inputs)
            loss = criterion(decoded, inputs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            
            # Logging
            running_loss += loss.item()
            if i % 100 == 99:
                logging.info('[epoch=%d, batch=%5d] loss: %.3f',
                             epoch + 1, i + 1, running_loss / 100)
                running_loss = 0.0
                
                # Save reconstructed images
                if i % 500 == 499:
                    vutils.save_image(torch.cat([inputs[:8], decoded[:8]], 0),
                                    os.path.join(args.checkpoint_path, f'reconstruction_e{epoch}_b{i}.png'),
                                    normalize=True, nrow=8)
        
        # Evaluate on test set
        metrics = test(net, testloader)
        
        # Save best model based on PSNR and SSIM
        if metrics['psnr'] > best_metrics['psnr'] or metrics['ssim'] > best_metrics['ssim']:
            best_metrics = metrics
            logging.info(f'New best model with PSNR: {metrics["psnr"]:.2f}dB, SSIM: {metrics["ssim"]:.4f}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, os.path.join(args.checkpoint_path, 'best_model.pt'))
        
        # Regular checkpoint saving
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_path, f'checkpoint_e{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, checkpoint_path)
            
        logging.info('Epoch %d finished, time: %.2fs', epoch + 1, timer() - time_start)

    logging.info(f'Finished Training. Best PSNR: {best_metrics["psnr"]:.2f}dB, Best SSIM: {best_metrics["ssim"]:.4f}')

if __name__ == '__main__':
    main() 