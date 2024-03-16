import numpy as np
import random
import warnings
import os
import time
from glob import glob
from skimage import color, io
from PIL import Image, ImageFilter, ImageOps

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, functional, Compose
import torch.nn.functional as F

def get_middlebury_images():
    root = "datasets/Middlebury/MiddEval3"
    with open(os.path.join(root, "official_train.txt"), 'r') as f:
        lines = f.read().splitlines()
    return sorted([os.path.join(root, 'trainingQ', f'{name}/im0.png') for name in lines])

def get_eth3d_images():
    return sorted(glob('datasets/ETH3D/two_view_training/*/im0.png'))

def get_kitti_images():
    return sorted(glob('datasets/KITTI/training/image_2/*_10.png'))

def transfer_color(image, style_mean, style_stddev):
    reference_image_lab = color.rgb2lab(image)
    reference_stddev = np.std(reference_image_lab, axis=(0,1), keepdims=True)# + 1
    reference_mean = np.mean(reference_image_lab, axis=(0,1), keepdims=True)

    reference_image_lab = reference_image_lab - reference_mean
    lamb = style_stddev/reference_stddev
    style_image_lab = lamb * reference_image_lab
    output_image_lab = style_image_lab + style_mean
    l, a, b = np.split(output_image_lab, 3, axis=2)
    l = l.clip(0, 100)
    output_image_lab = np.concatenate((l,a,b), axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        output_image_rgb = color.lab2rgb(output_image_lab) * 255
        return output_image_rgb

class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"

class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, yjitter=False, saturation_range=[0.6,1.4], gamma=[1,1,1,1]):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v': # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y1:y1+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
            
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow


    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow

class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False, yjitter=False, saturation_range=[0.7,1.3], gamma=[1,1,1,1]):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v': # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        return img1, img2, flow, valid


    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid



class TripletFlowAugmentor:
    '''
        reference: https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo/issues/22#issuecomment-1687742717
    '''
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, yjitter=False, saturation_range=[0.6,1.4], gamma=[1,1,1,1]):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

        # transform images into grayscale
        self.grayscale_prob = 0.1

    def color_transform(self, img0, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img0 = np.array(self.photo_aug(Image.fromarray(img0)), dtype=np.uint8)
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img0, img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img0, img1, img2 = np.split(image_stack, 3, axis=0)

        return img0, img1, img2

    def random_vertical_disp(self, inputs, angle, px, diff_angle=0, order=2, reshape=False):
        px2 = random.uniform(-px,px)
        angle2 = random.uniform(-angle,angle)

        image_center = (np.random.uniform(0,inputs[1].shape[0]),\
                             np.random.uniform(0,inputs[1].shape[1]))
        rot_mat = cv2.getRotationMatrix2D(image_center, angle2, 1.0)
        inputs[1] = cv2.warpAffine(inputs[1], rot_mat, inputs[1].shape[1::-1], flags=cv2.INTER_LINEAR)
        trans_mat = np.float32([[1,0,0],[0,1,px2]])
        inputs[1] = cv2.warpAffine(inputs[1], trans_mat, inputs[1].shape[1::-1], flags=cv2.INTER_LINEAR)
        return inputs

    # gt already filtered based on AO
    def spatial_transform(self, im1, im2, im3, gt=None, conf=None):

        # randomly sample scale
        ht, wd = im2.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale

        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            im1 = cv2.resize(im1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            im2 = cv2.resize(im2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            im3 = cv2.resize(im3, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            if gt is not None:
                gt = cv2.resize(gt, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST) * scale_x
                conf = cv2.resize(conf, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                tmp_left = im1[:, ::-1]
                tmp_center = im2[:, ::-1]
                tmp_right = im3[:, ::-1]
                im1 = tmp_right
                im2 = tmp_center
                im3 = tmp_left

                if gt is not None:
                    gt = gt[:, ::-1]
                    conf = conf[:, ::-1]

            if np.random.rand() < self.v_flip_prob: # v-flip
                im1 = im1[::-1, :]
                im2 = im2[::-1, :]
                im3 = im3[::-1, :]

                if gt is not None:
                    gt = gt[::-1, :]
                    conf = conf[::-1, :]

        # allow full size crops
        y0 = np.random.randint(2, im2.shape[0] - self.crop_size[0]-2)                  
        x0 = np.random.randint(2, im2.shape[1] - self.crop_size[1]-2)

        y1 = y0 + np.random.randint(-2, 2 + 1)

        im1_o = im1[:,:,:3][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]        
        im2_o = im2[:,:,:3][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        im3_o = im3[:,:,:3][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        im1_aug = im1[:,:,3:6][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]        
        im2_aug = im2[:,:,3:6][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        im3_aug = im3[:,:,3:6][y1:y1+self.crop_size[0], x0:x0+self.crop_size[1]]

        im1 = np.concatenate((im1_o,im1_aug),-1) 
        im2 = np.concatenate((im2_o,im2_aug),-1) 
        im3 = np.concatenate((im3_o,im3_aug),-1) 
  
        if gt is not None:
            gt = gt[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            conf = conf[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        angle=0;px=0
        if np.random.binomial(1,0.5):
            angle=0.1;px=3
            
        augmented = self.random_vertical_disp([im2[:,:,3:6], im3[:,:,3:6]], angle, px)

        # random occlusion to right image
        if np.random.rand() < self.eraser_aug_prob:
            sx = int(np.random.uniform(50, 100))
            sy = int(np.random.uniform(50, 100))
            cx = int(np.random.uniform(sx, im3.shape[0] - sx))
            cy = int(np.random.uniform(sy, im3.shape[1] - sy))
            augmented[1][cx - sx : cx + sx, cy - sy : cy + sy] = np.mean(
                np.mean(augmented[1], 0), 0
            )[np.newaxis, np.newaxis]

        im2 = np.concatenate((im2[:,:,:3],augmented[0]),-1) 
        im3 = np.concatenate((im3[:,:,:3],augmented[1]),-1) 

        return im1, im2, im3, gt, conf
        
    def __call__(self, im0, im1, im2, gt=None, conf=None):
        
        im0c, im1c, im2c = self.color_transform(im0, im1, im2)
        im0, im1, im2, gt, conf = self.spatial_transform(np.concatenate((im0,im0c),-1), np.concatenate((im1,im1c),-1), np.concatenate((im2,im2c),-1), gt, conf)

        if np.random.rand() < self.grayscale_prob:
            im1[:,:,3:6] = np.stack((cv2.cvtColor(im1[:,:,3:6], cv2.COLOR_BGR2GRAY),)*3, axis=-1)
            im2[:,:,3:6] = np.stack((cv2.cvtColor(im2[:,:,3:6], cv2.COLOR_BGR2GRAY),)*3, axis=-1)

        return {'im0': im0[:,:,:3], 'im1': im1[:,:,:3], 'im2': im2[:,:,:3], 'im0_aug': im0[:,:,3:6], 'im1_aug': im1[:,:,3:6], 'im2_aug': im2[:,:,3:6], 'disp': gt, 'conf': conf}






class CropAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, yjitter=False, saturation_range=[0.6,1.4], gamma=[1,1,1,1]):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)


        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow


    def __call__(self, img1, img2, flow):
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow






class FlowAugmentor_RTClean:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, yjitter=False, saturation_range=[0.6,1.4], gamma=[1,1,1,1]):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1_clean, img2_clean, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1_clean = cv2.resize(img1_clean, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2_clean = cv2.resize(img2_clean, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip
                img1_clean = img1_clean[:, ::-1]
                img2_clean = img2_clean[:, ::-1]
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
                tmp_clean = img1_clean[:, ::-1]
                img1_clean = img2_clean[:, ::-1]
                img2_clean = tmp_clean
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v': # v-flip
                img1_clean = img1_clean[::-1, :]
                img2_clean = img2_clean[::-1, :]
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1_clean = img1_clean[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2_clean = img2_clean[y1:y1+self.crop_size[0], x0:x0+self.crop_size[1]]
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y1:y1+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
            
            img1_clean = img1_clean[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2_clean = img2_clean[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1_clean, img2_clean, img1, img2, flow


    def __call__(self, img1, img2, flow):
        img1_clean = np.array(img1)
        img2_clean = np.array(img2)
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1_clean, img2_clean, img1, img2, flow = self.spatial_transform(img1_clean, img2_clean, img1, img2, flow)

        img1_clean = np.ascontiguousarray(img1_clean)
        img2_clean = np.ascontiguousarray(img2_clean)
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1_clean, img2_clean, img1, img2, flow



class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)

class GBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        
class PTrans:
    def __init__(self, num_patch=32, patch_r=32, num_view=4, cropscale=64):
        self.num_patch = num_patch
        self.patch_r = patch_r
        self.num_view = num_view
        # photometric augmentation params
        # self.photo_aug = Compose([ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 1.05

        self.aug_transform = transforms.Compose([
            # transforms.RandomResizedCrop(cropscale, scale=(0.25, 0.25), ratio=(1,1)),
            transforms.RandomResizedCrop(cropscale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
        ])

    def ptransform(self, img1, img2):
        """ Photometric augmentation """
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.aug_transform(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.aug_transform(Image.fromarray(img2)), dtype=np.uint8)
            # img1 = self.aug_transform(Image.fromarray(img1))
            # img2 = self.aug_transform(Image.fromarray(img2))
        # symmetric
        else:
            raise NameError("symmetric ptransform has not been implemented")
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def __call__(self, img1, img2, disp):
        # print(f'shape of img1 is {img1.shape}')
        # print(f'shape of img2 is {img2.shape}')
        # print(f'shape of disp is {disp.shape}')
        _, imgH, imgW = img1.shape
        patch_size = 2 * self.patch_r + 1

        # num_h = imgH // patch_size
        # num_w = imgW // patch_size
        # NUM_patch = num_h * num_w
        NUM_patch = self.num_patch
        # print(f'NUM_patch is {NUM_patch}')

        x_base = torch.linspace(0, 1, imgW).repeat(1, imgH, 1).type_as(img1)
        y_base = torch.linspace(0, 1, imgH).repeat(1, imgW, 1).transpose(1, 2).type_as(img1)
        # Apply shift in X direction
        x_shifts = -disp / imgW
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # print(f'shape of flow_field is {flow_field.shape}')

        index_h = torch.randint(patch_size//2+1, imgH-1-patch_size//2, (1, NUM_patch, 1, 1)) / imgH
        index_w = torch.randint(patch_size//2+1, imgW-1-patch_size//2, (1, NUM_patch, 1, 1)) / imgW
        index_center_left = torch.cat((index_w, index_h), dim=3) # N,1,1,2   NEED N, S, S, 2
        index_center_right = F.grid_sample(flow_field.permute(0, 3, 1, 2), 2 * index_center_left - 1, mode='nearest', padding_mode='border')
        index_center_left = index_center_left.transpose(0, 1)
        index_center_right = index_center_right.permute(2, 0, 3, 1)
        # print(f'shape of index_center_left is {index_center_left.shape}')
        # print(f'shape of index_center_right is {index_center_right.shape}')

        grid_h_array = torch.arange(-self.patch_r, self.patch_r+1).view(1, 1, -1, 1).repeat(1, patch_size, 1, 1).transpose(1, 2) / imgH
        grid_w_array = torch.arange(-self.patch_r, self.patch_r+1).view(1, 1, -1, 1).repeat(1, patch_size, 1, 1) / imgW
        # print(f'shape of grid_h_array is {grid_h_array.shape}')
        # print(f'shape of grid_w_array is {grid_w_array.shape}')
        # print(f'value of grid_h_array is {grid_h_array}')
        # print(f'value of grid_w_array is {grid_w_array}')
        grid = torch.cat((grid_w_array, grid_h_array), dim=3)
        # print(f'shape of grid is {grid.shape}')
        # print(f'value of grid is {grid}')
        # print(f'center of grid is {grid[:, 31, 31, :]}')
        # print(f'center of grid is {grid[:, 32, 32, :]}')
        # print(f'center of grid is {grid[:, 33, 33, :]}')

        index_patch_left = index_center_left + grid
        index_patch_left = index_patch_left.reshape(1, NUM_patch * (patch_size)**2, 1, 2)
        left_patches = F.grid_sample(img1.unsqueeze(0), 2 * index_patch_left - 1, mode='nearest', padding_mode='zeros')
        left_patches = left_patches.permute(2, 0, 3, 1).reshape(NUM_patch, patch_size, patch_size, 3).permute(0, 3, 1, 2)

        index_patch_right = index_center_right + grid
        index_patch_right = index_patch_right.reshape(1, NUM_patch * (patch_size)**2, 1, 2)
        right_patches = F.grid_sample(img2.unsqueeze(0), 2 * index_patch_right - 1, mode='nearest', padding_mode='zeros')
        right_patches = right_patches.permute(2, 0, 3, 1).reshape(NUM_patch, patch_size, patch_size, 3).permute(0, 3, 1, 2)

        augp_img1 = []
        augp_img2 = []
        # np.array(t.permute(1, 2, 0)).astype(np.uint8)[..., :3]
        for pid in range(left_patches.size(0)):
            x1 = np.array(left_patches[pid, :, :, :].permute(1, 2, 0)).astype(np.uint8)
            x2 = np.array(right_patches[pid, :, :, :].permute(1, 2, 0)).astype(np.uint8)
            x1views = []
            x2views = []
            for vid in range(self.num_view):
                aug_x1, aug_x2 = self.ptransform(x1, x2)
                aug_x1 = torch.from_numpy(aug_x1).permute(2, 0, 1).float()
                aug_x2 = torch.from_numpy(aug_x2).permute(2, 0, 1).float()
                x1views.append(aug_x1.unsqueeze(0))
                x2views.append(aug_x2.unsqueeze(0))
            x1views = torch.cat(x1views, dim=0)
            x2views = torch.cat(x2views, dim=0)
            augp_img1.append(x1views.unsqueeze(0))
            augp_img2.append(x2views.unsqueeze(0))
        augp_img1 = torch.cat(augp_img1, dim=0)
        augp_img2 = torch.cat(augp_img2, dim=0)
        return left_patches, right_patches, augp_img1, augp_img2