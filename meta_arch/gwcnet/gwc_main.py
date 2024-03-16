import torch.nn.functional as F
import torch.utils.data
from .submodules import *
import math
import logging

# from backbones.swin import swin_base_quarter, swin_large_quarter, swin_large_quarter_twostage
# from backbones.twins import twins_svt_large


try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


# def normalize_img(img0, img1):
#     # loaded images are in [0, 255]
#     # normalize by ImageNet mean and std
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
#     img0 = (img0 / 255. - mean) / std
#     img1 = (img1 / 255. - mean) / std

#     return img0, img1

def normalize_img(img):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
    img = (img / 255. - mean) / std

    return img


class ProjNeck(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super(ProjNeck, self).__init__()
        self.out_proj = nn.Linear(in_dim, out_dim)
        self.out_norm = norm_layer(out_dim)
    def forward(self, x):
        # assert len(x) == 1
        # x = x[0]
        bs, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.out_proj(x)
        x = self.out_norm(x)
        x = x.view(bs, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6



class GWCNet(nn.Module):
    def __init__(self, args):
        super(GWCNet, self).__init__()
        self.args = args
        self.maxdisp = args.maxdisp
        self.use_concat_volume = args.use_concat_volume
        print('use_concat_volume = {}'.format(self.use_concat_volume))

        self.num_groups = 40

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.ptrans = getattr(args, "ptrans", False)
        if self.ptrans:
            logging.info(f'ptrans mode is {self.ptrans}: set ptrans projection')
            hidden_dim = 320
            self.z_dim = 256
            self.norm_p = 2
            self.pool_func = nn.AdaptiveAvgPool2d(1)
            self.projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, self.z_dim))


        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def freeze_bn(self):
        pass
                    
    def cost_regularization(self, cost):
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.interpolate(cost0, scale_factor=4, mode='trilinear', align_corners=False)
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.interpolate(cost1, scale_factor=4, mode='trilinear', align_corners=False)
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.interpolate(cost2, scale_factor=4, mode='trilinear', align_corners=False)
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = F.interpolate(cost3, scale_factor=4, mode='trilinear', align_corners=False)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            
            return [-pred0.unsqueeze(1), -pred1.unsqueeze(1), -pred2.unsqueeze(1), -pred3.unsqueeze(1)]

        else:
            cost3 = self.classif3(out3)
            cost3 = F.interpolate(cost3, scale_factor=4, mode='trilinear', align_corners=False)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            
            return -pred3.unsqueeze(1)

    def forward(self, imgL, imgR, iters=None, flow_init=None, test_mode=False, return_inter=False, augp1=None, augp2=None):
        results = {}
        if return_inter:
            inter_results = {}
            
        imgL = (2 * (imgL / 255.0) - 1.0).contiguous()
        imgR = (2 * (imgR / 255.0) - 1.0).contiguous()

        # imgL = normalize_img(imgL)
        # imgR = normalize_img(imgR)

        if self.training and self.ptrans:
            augp1 = (2 * (augp1 / 255.0) - 1.0).contiguous()
            augp2 = (2 * (augp2 / 255.0) - 1.0).contiguous()
            Bs, NUM_P, NUM_V, _, _, _ = augp1.shape
            augps = torch.cat((augp1, augp2), dim=2)
            augps = augps.flatten(0, 2)
            with autocast(enabled=self.args.mixed_precision):
                feat_ps = self.feature_extraction(augps)["gwc_feature"]
                feat_ps = self.pool_func(feat_ps).squeeze(2).squeeze(2)
                z_ps = F.normalize(self.projection(feat_ps), p=self.norm_p).reshape(Bs, NUM_P, 2*NUM_V, self.z_dim)
                results.update(z_ps=z_ps)

        with autocast(enabled=self.args.mixed_precision):
            featL = self.feature_extraction(imgL)
            featR = self.feature_extraction(imgR)

            if return_inter:
                inter_results.update(match_left=featL["gwc_feature"])
                inter_results.update(match_right=featR["gwc_feature"])
                
            gwc_volume = build_gwc_volume(featL["gwc_feature"], featR["gwc_feature"], self.maxdisp // 4,
                                            self.num_groups)
            if self.use_concat_volume:
                concat_volume = build_concat_volume(featL["concat_feature"], featR["concat_feature"],
                                                    self.maxdisp // 4)
                volume = torch.cat((gwc_volume, concat_volume), 1)
            else:
                volume = gwc_volume
                    
            dispEsts = self.cost_regularization(volume)
        
        if test_mode:
            return None, dispEsts

        
        results.update(disp_preds=dispEsts)
        return results