import cv2
import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy
from runners.ncsn_runner_mcj_mms_segy import *

import os

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default='marmousi.yml',  help='Path to the config file') #celeba.yml
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--exp_my', type=str, default='D:\\', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='MmsSegyopen_4', help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.') #celeba
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--test', default=False, help='Whether to test the model')
    parser.add_argument('--sample', default=False, help='Whether to produce samples from the model')
    parser.add_argument('--fast_fid', default=False, help='Whether to do fast fid test')
    parser.add_argument('--resume_training', default=False, help='Whether to resume training')
    parser.add_argument('--resume_ckpt_id', type=int, default=0, help='resume_ckpt_id')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")

    # dataset
    parser.add_argument('--data_dir', default='/home/shendi_mcj/datasets/seismic/train', type=str,
                        help='path of train data')
    parser.add_argument('--siesmic_dir', default='/home/shendi_mcj/datasets/seismic/train', type=str,
                        help='path of train data')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    parser.add_argument('--patch_size', default=(32, 32), type=int, help='patch size of training data')
    parser.add_argument('--stride', default=(32, 32), type=int, help='the step size to slide on the data')
    parser.add_argument('--jump', default=3, type=int, help='the space between shot')
    parser.add_argument('--download', default=False, type=bool,
                        help='if you will download the dataset from the internet')
    parser.add_argument('--datasets', default=0, type=int,
                        help='the num of datasets you want be download,if download = True')
    parser.add_argument('--train_data_num', default=100000, type=int, help='the num of the train_data')
    parser.add_argument('--aug_times', default=0, type=int, help='Number of aug operations')
    parser.add_argument('--scales', default=[1], type=list, help='data scaling')
    parser.add_argument('--agc', default=False, type=int, help='Normalize each trace by amplitude')
    parser.add_argument('--verbose_', default=True, type=int, help='Whether to output the progress of data generation')
    parser.add_argument('--display', default=1000, type=int, help='interval for displaying loss')
    parser.add_argument('--cropped_patch_size', default=(32, 32), type=int, help='patch size of randomly cropped patch afer aug')
    parser.add_argument('--cropped_data_num', default=100000, type=int, help='the num of the randomly cropped patch afer aug')


    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.doc)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f,Loader = yaml.FullLoader)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    if not args.test and not args.sample and not args.fast_fid:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
            args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Image folder already exists. Overwrite? (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)
                else:
                    print("Output image folder exists. Program halted.")
                    sys.exit(0)

        elif args.fast_fid:
            os.makedirs(os.path.join(args.exp, 'fid_samples'), exist_ok=True)
            args.image_folder = os.path.join(args.exp, 'fid_samples', args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                overwrite = False
                if args.ni:
                    overwrite = False
                else:
                    response = input("Image folder already exists. \n "
                                     "Type Y to delete and start from an empty folder?\n"
                                     "Type N to overwrite existing folders (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")


    # args.resume_training = True
    # args.resume_ckpt_id = 195000
    args.siesmic_dir = '/home/shendi_mcj/datasets/seismic/marmousi/marmousi35'  # The folder where you put mat files, e.g., marmousi.mat
    args.data_dir = '/home/shendi_mcj/datasets/seismic/train' # The folder where you put segy/sgy files
    args.patch_size = (192, 192)
    args.stride = (128, 128)
    args.jump=3
    args.train_data_num = 100000
    args.agc = False
    args.cropped_patch_size= (128, 128)
    args.cropped_patch_num=100000

    config.data.seis_rescaled = True
    config.model.sigma_begin = 9
    config.model.sigma_end = 0.01
    config.model.num_classes = 1000 #500
    config.training.snapshot_freq=10000

    # print the arg pamameters
    for arg in vars(args):
        print('{:<15s}: {:s}'.format(arg, str(getattr(args, arg))))

    print(">" * 80)
    config_dict = copy.copy(vars(config))
    if not args.test and not args.sample and not args.fast_fid:
        del config_dict['tb_logger']
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    try:
        runner = NCSNRunner(args, config)
        if args.test:
            runner.test()
        elif args.sample:
            runner.sample()
        elif args.fast_fid:
            runner.fast_fid()
        else:
            runner.train()
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
