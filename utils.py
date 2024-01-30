import os
import sys
import torch
import shutil
import logging
import argparse
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable

def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)

def save(model, model_path):
  torch.save(model.state_dict(), model_path)

def load(model, model_path):
  model.load_state_dict(torch.load(model_path))

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, list(map(int, values.split(','))))

def get_network(args):
  if args.model_name in ('CifarNet', 'Cifarnet', 'cifarnet', 'Cifar', 'cifar'):
    from cifarnet import CifarNet_TREC
    return CifarNet_TREC(args.L, args.H, args.trec)
  elif args.model_name in ('SqueezeNet', 'Squeezenet', 'squeezenet', 'Squeeze', 'squeeze'):
    from squeezenet import squeezenet_trec
    return squeezenet_trec(args.L, args.H, args.trec)
  elif args.model_name in ('SqueezeNet_Complex_Bypass', 'Squeeze_complex_bypass', 'squeeze_complex_bypass', 'Squeeze_BP', 'Squeeze_bp','squeeze_bp'):
    from squeezenet_complex_bypass import SqueezeNet_TREC
    return SqueezeNet_TREC(args.L, args.H, args.bp_L, args.bp_H, args.trec, args.bp_trec)
