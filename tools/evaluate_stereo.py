from __future__ import print_function, division
import sys
import os
current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder

from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter
from utils import *

import json
from meta_arch import __models__, __losses__

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

# DIVIDE_FACTOR=28
DIVIDE_FACTOR=32

class ArgsContainer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False, logger=False, savename=None, global_step=-1, args={}, divide_factor=32):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))
        occ_mask = np.ascontiguousarray(occ_mask)

        if logger:
            # print(f'shape of image1 is {image1.shape}')
            image_outputs = {"imgL": image1.detach(), "imgR": image2.detach()}

        padder = InputPadder(image1.shape, divis_by=divide_factor)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)

        if logger:
            flow_gt_valid = flow_gt.detach()
            flow_gt_valid = torch.where(torch.isinf(flow_gt_valid), torch.full_like(flow_gt_valid, 0), flow_gt_valid)
            image, maxd = disp_to_color(np.array(-flow_gt_valid[0, :, :].cpu()))
            image_outputs['disp_gt'] = image
            image, _ = disp_to_color(np.array(-flow_pr.detach()[0, :, :].cpu()), maxd)
            image_outputs['disp_pred'] = image
            error_map = disp_error_map(-flow_pr, -flow_gt_valid, valid=flow_gt_valid<0)
            image_outputs['error_map'] = error_map
            image_outputs['occ_mask'] = occ_mask[None, :].astype(np.float32)
            image_outputs['sync'] = torch.zeros(flow_pr.shape).to(flow_pr.device).float()
            save_images(logger, savename + '_' + str(val_id), image_outputs, val_id)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt[0].reshape(-1) < 0) & (occ_mask.flatten() == 255)
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, split='mix', maxdisp=192, mixed_prec=False, logger=False, savename=None, global_step=-1, args={}, divide_factor=32):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, split=split, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        if logger:
            image_outputs = {"imgL": image1.detach(), "imgR": image2.detach()}

        padder = InputPadder(image1.shape, divis_by=divide_factor)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        if logger:
            flow_gt_valid = flow_gt.detach()
            flow_gt_valid = torch.where(torch.isinf(flow_gt_valid), torch.full_like(flow_gt_valid, 0), flow_gt_valid)
            image, maxd = disp_to_color(np.array(-flow_gt_valid[0, :, :].cpu()))
            image_outputs['disp_gt'] = image
            image, _ = disp_to_color(np.array(-flow_pr.detach()[0, :, :].cpu()), maxd)
            image_outputs['disp_pred'] = image
            error_map = disp_error_map(-flow_pr, -flow_gt_valid, valid=flow_gt_valid<0)
            image_outputs['error_map'] = error_map
            image_outputs['sync'] = torch.zeros(flow_pr.shape).to(flow_pr.device).float()
            save_images(logger, savename + '_' + str(val_id), image_outputs, val_id)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) > -maxdisp) & (flow_gt[0].reshape(-1) < 0)

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI {split} Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI-{split}: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {f'kitti-{split}-epe': epe, f'kitti-{split}-d1': d1}


@torch.no_grad()
def validate_things(model, iters=32, maxdisp=192, mixed_prec=False, logger=False, savename=None, global_step=-1, args={}, divide_factor=32):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=divide_factor)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe = epe.flatten()
        # val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) > -maxdisp)
        
        # assert not torch.isinf(flow_gt[0][val.bool()]).any()
        if(np.isnan(epe[val].mean().item())):
            continue
        out = (epe > 1.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation FlyingThings: %f, %f" % (epe, d1))
    return {'things-epe': epe, 'things-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, resolution='H', maxdisp=192, mixed_prec=False, logger=False, savename=None, global_step=-1, args={}, divide_factor=32):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, resolution=resolution)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32)

        if logger:
            image_outputs = {"imgL": image1.detach(), "imgR": image2.detach()}

        padder = InputPadder(image1.shape, divis_by=divide_factor)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        if logger:
            flow_gt_valid = flow_gt.detach()
            flow_gt_valid = torch.where(torch.isinf(flow_gt_valid), torch.full_like(flow_gt_valid, 0), flow_gt_valid)
            image, maxd = disp_to_color(np.array(-flow_gt_valid[0, :, :].cpu()))
            image_outputs['disp_gt'] = image
            image, _ = disp_to_color(np.array(-flow_pr.detach()[0, :, :].cpu()), maxd)
            image_outputs['disp_pred'] = image
            error_map = disp_error_map(-flow_pr, -flow_gt_valid, valid=flow_gt_valid<0)
            image_outputs['error_map'] = error_map
            image_outputs['occ_mask'] = occ_mask[None, :].astype(np.float32)
            image_outputs['sync'] = torch.zeros(flow_pr.shape).to(flow_pr.device).float()
            save_images(logger, savename + '_' + str(val_id), image_outputs, val_id)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) > -maxdisp) & (flow_gt[0].reshape(-1) < 0) & (occ_mask.flatten()==255)

        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}



@torch.no_grad()
def validate_booster(model, iters=32, resolution='Q', maxdisp=192, mixed_prec=False, logger=False, savename=None, global_step=-1, args={}, divide_factor=32):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Booster(aug_params, resolution=resolution)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        if logger:
            image_outputs = {"imgL": image1.detach(), "imgR": image2.detach()}

        padder = InputPadder(image1.shape, divis_by=divide_factor)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)


        if logger:
            flow_gt_valid = flow_gt.detach()
            flow_gt_valid = torch.where(torch.isinf(flow_gt_valid), torch.full_like(flow_gt_valid, 0), flow_gt_valid)
            image, maxd = disp_to_color(np.array(-flow_gt_valid[0, :, :].cpu()))
            image_outputs['disp_gt'] = image
            image, _ = disp_to_color(np.array(-flow_pr.detach()[0, :, :].cpu()), maxd)
            image_outputs['disp_pred'] = image
            error_map = disp_error_map(-flow_pr, -flow_gt_valid, valid=flow_gt_valid<0)
            image_outputs['error_map'] = error_map
            image_outputs['valid_gt'] = valid_gt[None, :] * 255.
            image_outputs['sync'] = torch.zeros(flow_pr.shape).to(flow_pr.device).float()
            save_images(logger, savename + '_' + str(val_id), image_outputs, val_id)


        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) > -maxdisp) & (flow_gt[0].reshape(-1) < 0)
        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Booster {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation Booster: EPE %f, D1 %f" % (epe, d1))
    return {'Booster-epe': epe, 'Booster-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help="config file to create model")
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    parser.add_argument('--logdir', default=False, help='the directory to save logs and checkpoints')
    args = parser.parse_args()

    with open(args.config,'r') as load_f:
        config_dic = json.load(load_f)
    config_dic = argparse.Namespace(**config_dic)
    args = argparse.Namespace(**vars(args), **vars(config_dic))
    print(args)

    if args.logdir:
        save_logger = SummaryWriter(args.logdir)
    else:
        save_logger = False

    model = __models__[args.model](args)
    model = torch.nn.DataParallel(model, device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(".ckpt")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    # use_mixed_precision = args.corr_implementation.endswith("_cuda")
    use_mixed_precision = False
    

    results = {}

    r_eth3d = validate_eth3d(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, logger=save_logger, savename='ETH3D_trainall', global_step=-1, args=args, divide_factor=DIVIDE_FACTOR)
    results.update(r_eth3d)

    r_midd_H = validate_middlebury(model, iters=args.valid_iters, resolution='H', mixed_prec=use_mixed_precision, logger=save_logger, savename='Midd_trainall', global_step=-1, divide_factor=DIVIDE_FACTOR)
    results.update(r_midd_H)

    r_2012 = validate_kitti(model, iters=args.valid_iters, split='2012', mixed_prec=use_mixed_precision, logger=save_logger, savename='KITTI2012_trainall', global_step=-1, divide_factor=DIVIDE_FACTOR)
    results.update(r_2012)

    r_2015 = validate_kitti(model, iters=args.valid_iters, split='2015', mixed_prec=use_mixed_precision, logger=save_logger, savename='KITTI2015_trainall', global_step=-1, divide_factor=DIVIDE_FACTOR)
    results.update(r_2015)

    r_booster = validate_booster(model, iters=args.valid_iters, resolution='Q', mixed_prec=use_mixed_precision, logger=save_logger, savename='Booster_trainall', global_step=-1, divide_factor=DIVIDE_FACTOR)
    results.update(r_booster)

    # r_things = validate_things(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, logger=save_logger, savename='SceneFlow_TestSub', global_step=-1, divide_factor=DIVIDE_FACTOR)
    # results.update(r_things)
    
    if save_logger:
        save_scalars(save_logger, 'results', results, -1)
