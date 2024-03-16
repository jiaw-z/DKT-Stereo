import torch.nn.functional as F
import torch
import torch.nn as nn

def loss_gwcnet(results, flow_gt, valid, args=None):
    flow_preds = results['disp_preds']
    max_flow = args.maxdisp
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(flow_preds, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[valid.bool()], flow_gt[valid.bool()], size_average=True))
    flow_loss = sum(all_losses)

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics, valid


