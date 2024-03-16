import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def context_upsample(depth_low, up_weights):
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = depth_low.shape
        
    depth_unfold = F.unfold(depth_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    depth_unfold = F.interpolate(depth_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    depth = (depth_unfold*up_weights).sum(1)
        
    return depth


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


def ssim(img1, img2, window_size=11):
    _, channel, h, w = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel)
    

def L1Loss(input, target):
    return (input - target).abs().mean()


def warp_disp(img, disp):
    '''
    Borrowed from: https://github.com/OniroAI/MonoDepth-PyTorch
    '''
    b, _, h, w = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(img)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :] / w
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='border')

    return output

def reprojection_error(img_left, img_right, disp=None, valid_mask=None, mask=None):
    b, _, h, w = img_left.shape
    if disp is not None:
        image_warped = warp_disp(img_right, -disp)
    else:
        image_warped = img_right

    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    # print(valid_mask.size())
    # print(image_warped.size())
    if mask is not None:
        # print(mask.size())
        valid_mask = valid_mask * mask
    # print(valid_mask.size())

    loss = 0.15 * L1Loss(image_warped * valid_mask, img_left * valid_mask) + \
           0.85 * (valid_mask * (1 - ssim(img_left, image_warped)) / 2).mean(dim=1)
    return loss


def unique(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


def uniqueness(disparity):
        # disparity = (disparity[:,:,:,0]).astype(np.uint8)
        disparity = (disparity).astype(np.uint8)
        batch = disparity.shape[0]
        height = disparity.shape[1]
        width = disparity.shape[2]
        coords = np.stack([np.stack([ np.arange(b*width*height + y*width, b*width*height + y*width + width) for y in range(height)], 0) for b in range(batch)], 0) - disparity
        array = np.reshape(coords, batch*height*width)
        output, index, _, _ = np.unique(array, return_index=True,return_inverse=True,return_counts=True)
        # print(output)
        # print(f'np_uni index is {index}')
        # print(f'len of np_uni index is {len(index)}')
        array *= 0
        array[index] = 1
        return np.reshape(array, (batch, height, width)).astype(np.float32)
        
def agreement(disparity, r, tau=1):
        # disparity = (disparity[:,:,:,0]).astype(np.uint8)
        disparity = (disparity).astype(np.uint8)
        height = disparity.shape[1]
        width = disparity.shape[2]
        batch = disparity.shape[0]
        disparity = np.pad(disparity, ((0,0),(r,r),(r,r)), 'constant')
        wind = (r*2+1)
        neighbors = np.stack([disparity[:,k//wind:k//wind+height,k%wind:k%wind+width] for k in range(wind**2)], -1)
        neighbors = np.delete(neighbors, wind**2//2, axis=-1)
        template = np.stack([disparity[:,r:r+height,r:r+width]]*(wind**2), -1)
        template = np.delete(template, wind**2//2, axis=-1)
        agreement = (np.sum( np.abs(template - neighbors) < tau, axis=-1, keepdims=False ) ).astype(np.float32)

        return agreement


def uniqueness_torch(disparity, device=None):
        # disparity = (disparity[:,:,:,0]).astype(np.uint8)
        # disparity = (disparity).astype(np.uint8)
        disparity = disparity.int()
        batch = disparity.shape[0]
        height = disparity.shape[1]
        width = disparity.shape[2]
        coords = torch.stack([torch.stack([ torch.arange(b*width*height + y*width, b*width*height + y*width + width) for y in range(height)], 0) for b in range(batch)], 0)
        coords = coords.to(disparity.device)
        coords = coords - disparity
        # array = torch.reshape(coords, batch*height*width)
        array = coords.flatten()
        # _, index, _, _ = torch.unique(array, return_index=True,return_inverse=True,return_counts=True)
        output, index = unique(array)
        # print(output)
        # print(f'torch_uni index is {index}')
        # print(f'len of torch_uni index is {len(index)}')
        array *= 0
        array[index] = 1
        return array.reshape(batch, height, width).float()
        # return torch.reshape(array, (batch, height, width)).float()


def agreement_torch(disparity, r, tau=1, device=None):
        # disparity = (disparity[:,:,:,0]).astype(np.uint8)
        # disparity = (disparity).astype(np.uint8)
        height = disparity.shape[1]
        width = disparity.shape[2]
        batch = disparity.shape[0]
        # disparity = np.pad(disparity, ((0,0),(r,r),(r,r)), 'constant')
        padding = (r, r, r, r, 0, 0)
        disparity = F.pad(disparity, padding, "constant", 0)
        # print(f'shape of padded disparity is {disparity.shape}')
        wind = (r*2+1)
        neighbors = torch.stack([disparity[:,k//wind:k//wind+height,k%wind:k%wind+width] for k in range(wind**2)], -1)
        # neighbors = np.delete(neighbors, wind**2//2, axis=-1)
        template = torch.stack([disparity[:,r:r+height,r:r+width]]*(wind**2), -1)
        # template = np.delete(template, wind**2//2, axis=-1)

        # neighbors = torch.tensor(neighbors).to(device)
        # template = torch.tensor(template).to(device)
        agreement = (torch.sum(torch.abs(template - neighbors) < tau, axis=-1, keepdims=False)).float()

        return agreement