import torch

def sequence_loss_raft(results, flow_gt, valid, loss_gamma=0.9, max_flow=700, args=None):
    """ Loss function defined over sequence of flow predictions """
    flow_preds = results['disp_preds']
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    # assert not torch.isinf(flow_gt[valid.bool()]).any()
    if torch.isinf(flow_gt[valid.bool()]).any():
        return None, None, None

    for i in range(n_predictions):
        # assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        if torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any():
            return None, None, None
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics, valid