import torch
import torch.nn.functional as F

def sequence_loss_pcvnet(results, disp_gt, valid,
                  max_disp=512, args=None):
    """ Loss function defined over sequence of disp predictions """
    gauss_num = args.gauss_num
    disp_refine, final_disp_preds, mu_preds, w_preds, sigma_preds = results['output_list']
    n_predictions = len(mu_preds)
    assert n_predictions >= 1
    disp_loss = 0.0

    # exclude extremely large displacements
    valid = (disp_gt < max_disp) & (valid[:, None].bool()) & (disp_gt >= 0)
    # print("mask_rate:", 1 - valid.float().mean())
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()
    N, C, H, W = mu_preds[0].shape
    i_weights = [0.4, 0.6, 0.8, 1, 1.2, 1.4]
    if (valid == False).all():
        disp_loss = 0
        metrics = {
            'epe': 0.,
            '1px': 1.,
            '3px': 1.,
            '5px': 1.,
        }
    else:
        for i in range(n_predictions):
            assert not torch.isnan(mu_preds[i]).any() and not torch.isinf(mu_preds[i]).any()
            assert not torch.isnan(w_preds[i]).any() and not torch.isinf(w_preds[i]).any()
            assert not torch.isnan(sigma_preds[i]).any() and not torch.isinf(sigma_preds[i]).any()
            assert not torch.isnan(final_disp_preds[i]).any() and not torch.isinf(final_disp_preds[i]).any()
            # split the segment
            mu_preds[i] = mu_preds[i].view(N // gauss_num, gauss_num, 1, H, W)

            w_preds[i] = w_preds[i].view(N // gauss_num, gauss_num, 1, H, W)
            sigma_preds[i] = sigma_preds[i].view(N // gauss_num, gauss_num, 1, H, W)
            w = w_preds[i]
            # print("mu%d:" % i, torch.mean(mu_preds[i], dim=[0, 2, 3, 4]).detach())
            # print("w%d:" % i, torch.mean(w, dim=[0, 2, 3, 4]).detach())
            # print("sigma%d:" % i, torch.mean(sigma_preds[i], dim=[0, 2, 3, 4]).detach())

            i_loss1 = (final_disp_preds[i] - disp_gt).abs()
            i_loss2 = torch.mean((mu_preds[i] - disp_gt[:, None]).abs(), dim=1)
            disp_loss += i_weights[i] * (
                    i_loss1.view(-1)[valid.view(-1)].mean() + i_loss2.view(-1)[valid.view(-1)].mean())
        disp_loss += 1.4 * F.smooth_l1_loss(disp_refine[valid], disp_gt[valid], size_average=True)

        epe_final = torch.abs(disp_refine - disp_gt)
        epe_final = epe_final.view(-1)[valid.view(-1)]

        epe = torch.abs(final_disp_preds[3] - disp_gt)
        epe = epe.view(-1)[valid.view(-1)]
        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
            'bad1': (epe > 1).float().mean().item(),
            'bad2': (epe > 2).float().mean().item(),
            'bad5': (epe > 5).float().mean().item(),

            'epe_final': epe_final.mean().item(),
            '1px_final': (epe_final < 1).float().mean().item(),
            '3px_final': (epe_final < 3).float().mean().item(),
            '5px_final': (epe_final < 5).float().mean().item(),
            'bad1_final': (epe_final > 1).float().mean().item(),
            'bad2_final': (epe_final > 2).float().mean().item(),
            'bad5_final': (epe_final > 5).float().mean().item(),
        }

    return disp_loss, metrics, valid