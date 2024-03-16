import torch
import torch.nn as nn


def SSIM(x, y, md=3):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    refl = nn.ReflectionPad2d(md)

    x = refl(x)
    y = refl(y)
    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist

def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1) 

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2) 

    base_grid = torch.stack([x_base, y_base], 1)
    return base_grid

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy

def smooth_grad(disp, image, alpha, order=1):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(disp)
    if order == 2:
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        dx, dy = dx2, dy2

    loss_x = weights_x[:, :, :, 1:] * dx[:, :, :, 1:].abs()
    loss_y = weights_y[:, :, 1:, :] * dy[:, :, 1:, :].abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.

def loss_smooth(disp, im1_scaled):
    func_smooth = smooth_grad
    loss = []
    loss += [func_smooth(disp, im1_scaled, 1, order=1)]
    return sum([l.mean() for l in loss])

def disp_warp(x, disp, r2l=False, pad='border', mode='bilinear', device='cuda'):
    B, _, H, W = x.size()
    offset = -1
    if r2l:
        offset = 1

    base_grid = mesh_grid(B, H, W).type_as(x)
    v_grid = norm_grid(base_grid + torch.cat((offset*disp,torch.zeros_like(disp)),1)) 
    x_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    mask = torch.autograd.Variable(torch.ones(x_recons.size())).to(device)
    mask = nn.functional.grid_sample(mask, v_grid)
    return x_recons, mask

def photometric_loss(im1_scaled, im1_recons, valid=None):
    loss = []
    loss += [0.15 * (im1_scaled - im1_recons).abs().mean(1, True)]
    loss += [0.85 * SSIM(im1_recons, im1_scaled).mean(1, True)]
    return sum([l for l in loss])

def trinocular_loss(disp, im1, im2, im3, uncertainty, valid=None):
    im2_recons_from_1, mask_12 = disp_warp(im1, disp, r2l=True)
    im2_recons_from_3, mask_23 = disp_warp(im3, disp, r2l=False)

    photometric_loss_12 = photometric_loss(im2, mask_12 * im2_recons_from_1)
    photometric_loss_23 = photometric_loss(im2, mask_23 * im2_recons_from_3)
    loss_warp, _ = torch.min(torch.cat((photometric_loss_12, photometric_loss_23), dim=1), dim=1)

    photometric_loss_1 = photometric_loss(im2, im1)
    photometric_loss_3 = photometric_loss(im2, im3)
    loss_2, _ = torch.min(torch.cat((photometric_loss_1, photometric_loss_3), dim=1), dim=1)

    automask = (loss_warp < loss_2) & (valid.squeeze(1).bool())
    # print(f'automask.shape is {automask.shape}')
    #  & (valid.bool())
    loss = (loss_warp * uncertainty)[automask]

    return loss.mean()

def binocular_loss(disp, im1, im2, uncertainty):
    im1_recons, _ = disp_warp(im2, disp, r2l=False)

    loss_warp = photometric_loss(im1, im1_recons).squeeze()
    loss_2 = photometric_loss(im2, im1).squeeze()

    automask = loss_warp < loss_2
    loss = (loss_warp * uncertainty)[automask]

    return loss[valid.bool()].mean()

def image_loss(disp, im1, im2, im3, uncertainty, trinocular=True, valid=None):
    if trinocular:
        return trinocular_loss(disp, im1, im2, im3, uncertainty, valid=valid)
    else:
        return binocular_loss(disp, im2, im3, uncertainty, valid=valid)

def ns_loss(pred_disps, target_disp, conf, im0, im1, im2, trinocular_loss=True, alpha_disp_loss=1.0, alpha_photometric=0.1, conf_threshold=0.5, max_flow=512):
    '''
        pred_disps and target_disp are negative here
    '''
    target_disp = target_disp.unsqueeze(1)
    conf = conf * (target_disp.squeeze(1) < 0).float()

    valid = (conf > conf_threshold)
    # print(f'target_disp.shape is {target_disp.shape}')
    # print(f'valid.shape is {valid.shape}')

    mag = torch.sum(target_disp**2, dim=1).sqrt()
    # print(f'mag.shape is {mag.shape}')
    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    # print(f'valid.shape is {valid.shape}')
    assert valid.shape == target_disp.shape, [valid.shape, target_disp.shape]
    assert not torch.isinf(target_disp[valid.bool()]).any()

    n_predictions = len(pred_disps)
    loss_gamma = 0.9
    

    disp_loss = 0.0
    photometric_loss = 0.0

    for i in range(n_predictions):
        assert not torch.isnan(pred_disps[i]).any() and not torch.isinf(pred_disps[i]).any()

        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        
        disp_diff = torch.abs(pred_disps[i] - target_disp)
        # disp_loss += i_weight * (disp_diff * conf.unsqueeze(1)).mean()
        disp_loss += i_weight * (disp_diff * conf.unsqueeze(1))[valid.bool()].mean()
        
        if alpha_photometric != 0.:
            photometric_loss += i_weight * image_loss(pred_disps[i], im0, im1, im2, 1 - conf, trinocular_loss, valid=valid)
        else:
            photometric_loss = torch.zeros_like(disp_loss)

    loss = alpha_disp_loss * disp_loss + alpha_photometric * photometric_loss

    epe = torch.sum((pred_disps[-1] - target_disp)**2, dim=1).sqrt()
    epe = epe.view(-1)[(valid).view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return loss, metrics, valid.squeeze(1)