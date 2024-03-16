from __future__ import print_function, division
import sys
import os
current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

import random

from tools.evaluate_stereo import *
import core.stereo_datasets as datasets

import torch.nn.functional as F
from utils.confidence import *

import json
from meta_arch import __models__, __losses__

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100
    def __init__(self, model, scheduler, save_root=None):
        self.model = model
        self.scheduler = scheduler
        self.save_root = save_root
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=self.save_root)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.save_root)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.save_root)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = __models__[args.model](args)
    loss_func = __losses__[args.loss_func]
    model = nn.DataParallel(model)
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler, args.save_dir)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 5000

    scaler = GradScaler(enabled=args.mixed_precision)

    # results = {}
    # results.update(validate_eth3d(model.module,iters=args.valid_iters))
    # results.update(validate_middlebury(model.module, iters=args.valid_iters, split='H'))
    # results.update(validate_kitti(model.module, iters=args.valid_iters, split='2012'))
    # results.update(validate_kitti(model.module, iters=args.valid_iters, split='2015'))
    # results.update(validate_booster(model.module, iters=args.valid_iters))
    # results.update(validate_things(model.module, iters=args.valid_iters))
    # logger.write_dict(results)

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:
        # for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
        for i_batch, (_, data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            # image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]
            image1_clean = data_blob['img1_clean'].cuda()
            image2_clean = data_blob['img2_clean'].cuda()
            image1 = data_blob['img1'].cuda()
            image2 = data_blob['img2'].cuda()
            disp_gt = data_blob['flow'].cuda()
            valid = data_blob['valid'].cuda()

            assert model.training
            results = model(image1, image2, iters=args.train_iters)
            disp_preds = results['disp_preds']
            assert model.training

            loss, metrics, valid_final = loss_func(results, disp_gt, valid, args=args)
            if loss is None:
                global_batch_num += 1
                continue
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % 50 == 0:
                image_outputs = {"image1": image1, "image2": image2}
                image_outputs['image1_clean'] = image1_clean
                image_outputs['image2_clean'] = image2_clean
                image, maxd_gt = disp_to_color(np.array(-disp_gt.squeeze(1)[0, :, :].cpu()))
                image_outputs['disp_gt'] = image
                image, _ = disp_to_color(np.array(-disp_preds[-1].detach().squeeze(1)[0, :, :].cpu()), maxd_gt)
                image_outputs['disp_pred'] = image
                image_outputs['error_map'] = disp_error_map(-disp_preds[-1].squeeze(1), -disp_gt.squeeze(1), valid=valid_final.squeeze(1))
                image_outputs['valid'] = valid_final.float() * 255.

                image_outputs["zzz_for_syn"] = torch.zeros(disp_gt.shape).to(disp_gt.device).float()
                save_images(logger.writer, '_train', image_outputs, total_steps)


            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path(args.save_dir + '/%d_%s.pth' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save({'total_steps': total_steps + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, save_path)

                results = {}
                results.update(validate_eth3d(model.module,iters=args.valid_iters))
                results.update(validate_middlebury(model.module, iters=args.valid_iters, split='H'))
                results.update(validate_kitti(model.module, iters=args.valid_iters, split='2012'))
                results.update(validate_kitti(model.module, iters=args.valid_iters, split='2015'))
                results.update(validate_booster(model.module, iters=args.valid_iters))
                # results.update(validate_things(model.module, iters=args.valid_iters))
                logger.write_dict(results)

                model.train()
                model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            save_path = Path(args.save_dir + '/%d_epoch_%s.pth.gz' % (total_steps + 1, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    PATH = args.save_dir + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help="config file to create model")
    parser.add_argument('--name', default='model', help="name your experiment")
    parser.add_argument('--save_dir', default='runs/debug', help="name your save_root")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # # Architecure choices backbone_type
    # parser.add_argument('--backbone_type', type=str, default='default', help="type of backbone")
    # parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda", "cosine"], default="reg", help="correlation volume implementation")
    # parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    # parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    # parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    # parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    # parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    # parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    # parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    # parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    with open(args.config,'r') as load_f:
        config_dic = json.load(load_f)
    config_dic = argparse.Namespace(**config_dic)
    args = argparse.Namespace(**vars(args), **vars(config_dic))
    print(args)

    # torch.manual_seed(1234)
    # np.random.seed(1234)

    seed_everything(1234)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    train(args)