from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable, Function
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import cm
from typing import Optional, Sequence


def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


def gen_error_colormap():
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols


error_colormap = gen_error_colormap()




def disp_error_map(D_est_tensor, D_gt_tensor, abs_thres=3., rel_thres=0.05, dilate_radius=1, valid=None):
    D_gt_np = D_gt_tensor.detach().cpu().numpy()
    D_est_np = D_est_tensor.detach().cpu().numpy()
    B, H, W = D_gt_np.shape
    # valid mask
    # mask = D_gt_np > 0
    mask = np.ones(D_gt_np.shape, dtype=bool)
    if not valid is None:
        valid = valid.detach().cpu().numpy()
        valid = valid >= 0.5
        # print(f'shape of valid is {valid.shape}')
        mask = mask & valid
        # print(mask)
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = np.minimum(error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres)
    # get colormap
    cols = error_colormap
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]

    return torch.from_numpy(np.ascontiguousarray(error_image.transpose([0, 3, 1, 2]))) * 255.


class disp_error_image_func(Function):
    @staticmethod
    def forward(self, D_est_tensor, D_gt_tensor, abs_thres=3., rel_thres=0.05, dilate_radius=1):
        D_gt_np = D_gt_tensor.detach().cpu().numpy()
        D_est_np = D_est_tensor.detach().cpu().numpy()
        B, H, W = D_gt_np.shape
        # valid mask
        mask = D_gt_np > 0
        # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
        error = np.abs(D_gt_np - D_est_np)
        error[np.logical_not(mask)] = 0
        error[mask] = np.minimum(error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres)
        # get colormap
        cols = error_colormap
        # create error image
        error_image = np.zeros([B, H, W, 3], dtype=np.float32)
        for i in range(cols.shape[0]):
            error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
        # TODO: imdilate
        # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
        error_image[np.logical_not(mask)] = 0.
        # show color tag in the top-left cornor of the image
        for i in range(cols.shape[0]):
            distance = 20
            error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]

        return torch.from_numpy(np.ascontiguousarray(error_image.transpose([0, 3, 1, 2])))
    
    @staticmethod
    def backward(self, grad_output):
        return None



from matplotlib.pyplot import MultipleLocator
class matplotlib_tools:
    def __init__(self):
        self.width= 6
        self.height= 4
        self.dpi= 128
        self.fig_convert = plt.figure(figsize=(self.width, self.height), dpi=self.dpi)
        self.axes_convert = self.fig_convert.add_axes([0.16, 0.15, 0.75, 0.75])

    def plot(self, x, y, x_step=None):
        self.axes_convert.cla()
        self.axes_convert.plot(x, y)
        if x_step is not None:
            x_major_locator = MultipleLocator(x_step)
            self.axes_convert.xaxis.set_major_locator(x_major_locator)

        self.fig_convert.canvas.draw()
        fig_str = self.fig_convert.canvas.tostring_rgb()
        data = np.frombuffer(fig_str, dtype=np.uint8).reshape((self.height*self.dpi, -1, 3)) / 255.0
        return data
    def bar(self, x, y, title='debug'):
        FONTSIZE=12
        xlabels = ["%.1d" % t for t in x]
        xlabels[-1] = '+'
        y.append(-np.inf)
        self.axes_convert.cla()
        self.axes_convert.bar(x, y, width=9, align='edge', color='green')
        for a,b in zip(x, y):
            self.axes_convert.text(a,b,'%.2f' % b,  fontdict={'fontsize': FONTSIZE})
        self.axes_convert.set_xticks(x)
        self.axes_convert.set_xticklabels(xlabels, fontdict={'fontsize': FONTSIZE})
        self.axes_convert.set_xlabel("Depth", fontdict={'fontsize': FONTSIZE})
        self.axes_convert.set_ylabel("DepthEPE(m)", fontdict={'fontsize': FONTSIZE})
        self.axes_convert.set_title(title, fontdict={'fontsize': FONTSIZE})

        self.fig_convert.canvas.draw()
        fig_str = self.fig_convert.canvas.tostring_rgb()
        data = np.frombuffer(fig_str, dtype=np.uint8).reshape((self.height*self.dpi, -1, 3)) / 255.0
        return data




def disp_map(disp):
    """
    Based on color histogram, convert the gray disp into color disp map.
    The histogram consists of 7 bins, value of each is e.g. [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    Accumulate each bin, named cbins, and scale it to [0,1], e.g. [0.114, 0.299, 0.413, 0.587, 0.701, 0.886, 1.0]
    For each value in disp, we have to find which bin it belongs to
    Therefore, we have to compare it with every value in cbins
    Finally, we have to get the ratio of it accounts for the bin, and then we can interpolate it with the histogram map
    For example, 0.780 belongs to the 5th bin, the ratio is (0.780-0.701)/0.114,
    then we can interpolate it into 3 channel with the 5th [0, 1, 0] and 6th [0, 1, 1] channel-map
    Inputs:
        disp: numpy array, disparity gray map in (Height * Width, 1) layout, value range [0,1]
    Outputs:
        disp: numpy array, disparity color map in (Height * Width, 3) layout, value range [0,1]
    """
    map = np.array([
        [0, 0, 0, 114],
        [0, 0, 1, 185],
        [1, 0, 0, 114],
        [1, 0, 1, 174],
        [0, 1, 0, 114],
        [0, 1, 1, 185],
        [1, 1, 0, 114],
        [1, 1, 1, 0]
    ])
    # grab the last element of each column and convert into float type, e.g. 114 -> 114.0
    # the final result: [114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0]
    bins = map[0:map.shape[0] - 1, map.shape[1] - 1].astype(float)

    # reshape the bins from [7] into [7,1]
    bins = bins.reshape((bins.shape[0], 1))

    # accumulate element in bins, and get [114.0, 299.0, 413.0, 587.0, 701.0, 886.0, 1000.0]
    cbins = np.cumsum(bins)

    # divide the last element in cbins, e.g. 1000.0
    bins = bins / cbins[cbins.shape[0] - 1]

    # divide the last element of cbins, e.g. 1000.0, and reshape it, final shape [6,1]
    cbins = cbins[0:cbins.shape[0] - 1] / cbins[cbins.shape[0] - 1]
    cbins = cbins.reshape((cbins.shape[0], 1))

    # transpose disp array, and repeat disp 6 times in axis-0, 1 times in axis-1, final shape=[6, Height*Width]
    ind = np.tile(disp.T, (6, 1))
    tmp = np.tile(cbins, (1, disp.size))

    # get the number of disp's elements bigger than  each value in cbins, and sum up the 6 numbers
    b = (ind > tmp).astype(int)
    s = np.sum(b, axis=0)

    bins = 1 / bins

    # add an element 0 ahead of cbins, [0, cbins]
    t = cbins
    cbins = np.zeros((cbins.size + 1, 1))
    cbins[1:] = t

    # get the ratio and interpolate it
    disp = (disp - cbins[s]) * bins[s]
    disp = map[s, 0:3] * np.tile(1 - disp, (1, 3)) + map[s + 1, 0:3] * np.tile(disp, (1, 3))

    return disp




def disp_to_color(disp, max_disp=None):
    """
    Transfer disparity map to color map
    Args:
        disp (numpy.array): disparity map in (Height, Width) layout, value range [0, 255]
        max_disp (int): max disparity, optionally specifies the scaling factor
    Returns:
        disparity color map (numpy.array): disparity map in (Height, Width, 3) layout,
            range [0,255]
    """
    # grab the disp shape(Height, Width)
    # @@@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # max_disp=192

    h, w = disp.shape

    # if max_disp not provided, set as the max value in disp
    if max_disp is None:
        max_disp = np.max(disp)
        # print(f'shape of disp = {disp.shape}')
        # print(f'max_disp = {max_disp}')
        # print(disp)

    # # scale the disp to [0,1] by max_disp
    # disp = disp / max_disp

    # # reshape the disparity to [Height*Width, 1]
    # disp = disp.reshape((h * w, 1))

    # # convert to color map, with shape [Height*Width, 3]
    # disp = disp_map(disp)

    # # convert to RGB-mode
    # disp = disp.reshape((1, h, w, 3)).transpose(0, 3, 1, 2)
    # disp = disp * 255.
    # # disp = disp

    # disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    disp = plt.Normalize(0, max_disp)(disp)
    disp = cm.jet(disp)[:, :, 0:3].reshape((1, h, w, 3)).transpose(0, 3, 1, 2)
    # print(f'shape of disp is {disp.shape}')
    disp = disp * 255.

    return disp, max_disp




def _compute_point_distance(disp_pred, disp_gt, f, b):
    """Compute point distance in 3D space."""
    inv_disp_pred = 1.0 / (np.abs(disp_pred) + 1e-6)
    inv_disp_gt = 1.0 / (np.abs(disp_gt) + 1e-6)

    # make meshgrid
    ht, wd = disp_pred.shape[:2]
    gridx, gridy = np.meshgrid(np.arange(wd), np.arange(ht))
    gridx = gridx - (wd - 1.0) / 2.0
    gridy = gridy - (ht - 1.0) / 2.0

    # compute 3d points for pred
    pred_x = gridx * b * inv_disp_pred
    pred_y = gridy * b * inv_disp_pred
    pred_z = b * f * inv_disp_pred

    # compute 3d points for gt
    gt_x = gridx * b * inv_disp_gt
    gt_y = gridy * b * inv_disp_gt
    gt_z = b * f * inv_disp_gt

    # compute L2 distance in 3d space
    delta_x = pred_x - gt_x
    delta_y = pred_y - gt_y
    delta_z = pred_z - gt_z
    l2_dist = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    return l2_dist


def epe_on_depth(
    disp_pred: np.ndarray,
    disp_gt: np.ndarray,
    valid_gt: np.ndarray,
    img_meta: dict,
    depth_bins: Optional[Sequence[float]] = None,
    max_visible_depth: Optional[float] = 200.0,
    ):
    """
    Args:
        disp_pred (np.ndarray): Disparity prediction.
        disp_gt (np.ndarray): Disparity ground truth.
        valid_gt (np.ndarray): Valid mask of ground truth.
        img_meta (dict): Meta information of image.
            contain keys: baseline, focal
        depth_bins (Optional[Sequence[float]]): Depth bins for evaluation.
            e.g. [0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-100, 100+]
            if the last element is not np.inf, add np.inf to the end.
        max_visible_depth (Optional[float]): Maximum visible depth for evaluation.
            Default: 200.0, avoid outliers for valid region.
    Returns:
        dict: Evaluation metrics.
    """
    assert "baseline" in img_meta
    assert "focal" in img_meta
    metrics = dict()

    b = img_meta["baseline"]
    f = img_meta["focal"]
    val = valid_gt >= 0.5
    # print(val)
    metrics["NumGT@all"] = np.sum(val)

    if disp_pred.ndim == 3:
        disp_pred = disp_pred[:, :, 0]
    if disp_gt.ndim == 3:
        disp_gt = disp_gt[:, :, 0]
    metrics["NumPred@all"] = np.sum(disp_pred > 0.0)
    disp_diff = np.abs(disp_pred - disp_gt)

    # compute depth error
    depth_pred = b * f / (np.abs(disp_pred) + 1e-6)
    depth_gt = b * f / (np.abs(disp_gt) + 1e-6)
    depth_diff = np.abs(depth_pred - depth_gt)

    # compute 3d point distance
    point_dist = _compute_point_distance(disp_pred, disp_gt, f, b)

    # average over valid pixels
    # outlier_mask = depth_diff[val] > max_visible_depth
    # metrics["NumOutlier@all"] = np.sum(outlier_mask)
    # clip depth error to 200m to avoid outliers
    if np.sum(val) == 0:
        metrics["DepthEPE@all"] = 0.0
        metrics["DispEPE@all"] = 0.0
        metrics["DepthRel@all"] = 0.0
        metrics["3DEPE@all"] = 0.0
    else:
        metrics["DispEPE@all"] = np.mean(disp_diff[val])
        metrics["DepthEPE@all"] = np.mean(depth_diff[val].clip(0, max_visible_depth))
        depth_rel = depth_diff[val].clip(0, max_visible_depth) / depth_gt[val]
        metrics["DepthRel@all"] = np.mean(depth_rel) * 100
        metrics["3DEPE@all"] = np.mean(point_dist[val].clip(0, max_visible_depth))

    if depth_bins is not None:
        # if depth_bins is not end with np.inf, add np.inf to the end
        if depth_bins[-1] != np.inf:
            depth_bins.append(np.inf)

        # depth_bins: e.g.
        #   [0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-100, 100+]
        # compute mean epe for each depth bin
        epe_depth_distribution = []
        val_depth_distribution = []
        for i in range(len(depth_bins) - 1):
            min_depth = depth_bins[i]
            max_depth = depth_bins[i + 1]
            name = "{}~{}".format(min_depth, max_depth)

            mask = np.logical_and(depth_gt > min_depth, depth_gt <= max_depth)
            mask = np.logical_and(mask, val)
            metrics["NumGT@{}".format(name)] = np.sum(mask)
            mask_pred = np.logical_and(depth_pred > min_depth, depth_pred <= max_depth)
            metrics["NumPred@{}".format(name)] = np.sum(mask_pred)
            # outlier_mask = depth_diff[mask] > max_visible_depth
            # metrics["NumOutlier@{}".format(name)] = np.sum(outlier_mask)

            if np.sum(mask) == 0:
                metrics["DepthEPE@{}".format(name)] = 0.0
                metrics["DispEPE@{}".format(name)] = 0.0
                metrics["DepthRel@{}".format(name)] = 0.0
                metrics["3DEPE@{}".format(name)] = 0.0
                epe_depth_distribution.append(0.0)
                val_depth_distribution.append(0)
                continue

            DispEPE = np.mean(disp_diff[mask])
            metrics["DispEPE@{}".format(name)] = DispEPE
            metrics["DepthEPE@{}".format(name)] = np.mean(
                depth_diff[mask].clip(0, max_visible_depth)
            )
            # relative depth error, delta_z / z_gt
            depth_rel = depth_diff[mask].clip(0, max_visible_depth) / depth_gt[mask]
            metrics["DepthRel@{}".format(name)] = np.mean(depth_rel) * 100
            metrics["3DEPE@{}".format(name)] = np.mean(
                point_dist[mask].clip(0, max_visible_depth)
            )
            
            epe_depth_distribution.append(np.mean(depth_diff[mask].clip(0, max_visible_depth)))
            val_depth_distribution.append(1)
        
        metrics['epe_depth_distribution'] = epe_depth_distribution
        metrics['val_depth_distribution'] = val_depth_distribution
        

    return metrics






"""Convert disparity map to point cloud and save to ply file for visualization.
Usage:
    data prepare: disparity map, calibration file, [optional] image
    img format: [H, W, 3] NOTE: in BGR order
    disp format: [H, W] in float32
    calib format: yml file with Q matrix
    save_path: save path of point cloud, default is None

    depth, points, colors = disp_to_plypc(img, disp, calib_path, save_path=None)
"""


# __all__ = ["disp_to_plypc", "disp_to_depth"]


def _load_calib_from_yml(calib_path: str):
    assert calib_path.endswith(".yml") or calib_path.endswith(".yaml")

    ymlfile = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    camera_kp = ymlfile.getNode("Q").mat().reshape([4, 4])
    ymlfile.release()

    f = camera_kp[2, 3]
    b = 1.0 / camera_kp[3, 2]
    cx = -camera_kp[0, 3]
    cy = -camera_kp[1, 3]

    return f, b, cx, cy


def _get_camK(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def disp_to_plypc(
    disp,
    calib_path,
    img=None,
    min_depth=0.0,
    max_depth=200.0,
    save_path: str = None,
):
    """Remap disparity map to point cloud and save to ply file.
    Args:
        disp (np.ndarray): Disparity map in [H, W]
        calib_path (str): Calibration file path, yml file
        img (np.ndarray): Image in [H, W, 3], default is None,
            if None, using white image as default
            default channel order is RGB color
        min_depth (float): Minimum depth, default 0. (m)
        max_depth (float): Maximum depth, default 100. (m)
        save_path (str): Save path of point cloud, default is None
    Returns:
        np.ndarray: Depth map in [H, W, 1]
        np.ndarray: Points in camera coordinate in [3, N]
        np.ndarray: Colors in [3, N]
    """
    # check input
    assert (
        isinstance(disp, np.ndarray) and disp.ndim == 2
    ), "disp should be a 2D numpy array"

    if img is None:
        # using white image as default
        img_h, img_w = disp.shape[:2]
        img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    # load camrea intrinsics
    f, b, cx, cy = _load_calib_from_yml(calib_path)
    intr_martix = _get_camK(f, cx, cy)

    # disparity to depth
    depth = disp_to_depth(
        disp,
        focus=f,
        baseline=b,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    y_index, x_index = np.nonzero(depth)
    ones = np.ones(len(x_index))
    pix_coords = np.stack([x_index, y_index, ones], axis=0)
    normalize_points = np.dot(np.linalg.inv(intr_martix), pix_coords)
    points = normalize_points * depth[y_index, x_index]
    colors = img[y_index, x_index].transpose(1, 0)

    # save points to ply file
    if save_path is not None:
        generate_pointcloud_ply(points, colors, save_path)

    return depth, points, colors


def disp_to_depth(disp, focus, baseline, min_depth=0.0, max_depth=100):
    """Convert disparity to depth map.
    Args:
        disp (np.ndarray): Disparity map in [H, W]
        focus (float): Focal length, default 721.5377 (px)
        baseline (float): Stereo baseline, default 0.532725 (m)
        min_depth (float): Minimum depth, default 0. (m)
        max_depth (float): Maximum depth, default 100. (m)
    Returns:
        np.ndarray: Depth map in [H, W, 1]
    """
    assert (
        isinstance(disp, np.ndarray) and disp.ndim == 2
    ), "disp should be a 2D numpy array"

    u, v = np.meshgrid(range(disp.shape[1]), range(disp.shape[0]))
    u = u.flatten()
    v = v.flatten()
    depth = focus * baseline / (disp.flatten().astype("float") + 1e-9)

    depth[depth < min_depth] = 0.0
    depth[depth > max_depth] = 0.0

    return depth.reshape(disp.shape)



def generate_pointcloud_ply(xyz, color, pc_file):
    """Generate point cloud in ply format.
    Args:
        xyz (np.ndarray): Points in camera coordinate in [3, N], float32
        color (np.ndarray): Colors in [3, N], uint8 channel order: RGB
        pc_file (str): Save path of point cloud
    Returns:
        None
    """
    assert (
        isinstance(xyz, np.ndarray) and xyz.ndim == 2 and xyz.shape[0] == 3
    ), "xyz should be a 2D numpy array"
    assert (
        isinstance(color, np.ndarray) and color.ndim == 2 and color.shape[0] == 3
    ), "color should be a 2D numpy array"
    assert (
        xyz.shape[1] == color.shape[1]
    ), "xyz and color should have the same number of points"

    df = np.zeros((6, xyz.shape[1]))
    df[0], df[1], df[2] = xyz[0], xyz[1], xyz[2]
    df[3], df[4], df[5] = color[0], color[1], color[2]

    def _to_ply_str(x):
        def float_formatter(x):
            return "%.4f" % x

        return "{} {} {} {} {} {} 0\n".format(
            float_formatter(x[0]),
            float_formatter(x[1]),
            float_formatter(x[2]),
            int(x[3]),
            int(x[4]),
            int(x[5]),
        )

    points = [_to_ply_str(i) for i in df.T]
    # save points to ply file
    with open(pc_file, "w") as f:
        f.write(
            """ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            """
            % (len(points), "".join(points))
        )
