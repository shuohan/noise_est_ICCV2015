#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-07 15:13:19

import sys
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_gradient_magnitude
import matplotlib.pyplot as plt


def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')

    
    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))


def im2patch_mask(im, mask, pch_size, stride=1, output_indices=False):
    threshold = 1
    im_patches = im2patch(im, pch_size, stride)
    mask_patches = im2patch(mask.astype(bool), pch_size, stride)
    tmp1 = np.sum(mask_patches, axis=(0, 1, 2))
    tmp2 = threshold * np.prod(im_patches.shape[:3])
    indices = tmp1 >= tmp2
    im_patches = im_patches[:, :, :, indices]
    if output_indices:
        return im_patches, indices
    else:
        return im_patches


def im2patch_mask_grad(im, mask, pch_size, stride=1):
    portion = 1/2

    im_patches, indices = im2patch_mask(im, mask, pch_size, stride, True)
    grad = gaussian_gradient_magnitude(im, 3)
    grad_patches = im2patch(grad, pch_size, stride)
    grad_patches = grad_patches[:, :, :, indices]
    flat = np.sum(np.abs(grad_patches), axis=(0, 1, 2))
    
    order = np.argsort(flat)
    order = order[: int(len(order) * portion)]
    im_patches = im_patches[:, :, :, order]
    return im_patches


def im2double(im):
    '''
    Input:
        im: numpy uint format image, RGB or Gray, 
    '''

    im = im.astype(np.float)
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())

    out = (im - min_val) / (max_val - min_val)

    return out


def calc_fg_mask(image, sigma=5):
    num_labels = 2

    blur_image = gaussian_filter(image.squeeze(), sigma)
    features = blur_image.flatten()[..., None]

    kmeans = KMeans(n_clusters=num_labels)
    seg = kmeans.fit_predict(features)

    # foreground has greater intensities
    mean_int = [np.mean(features[seg==l, 0]) for l in range(num_labels)]
    order = np.argsort(mean_int)
    seg = order[seg] > 0
    seg = seg.reshape(image.shape)

    # import matplotlib.pyplot as plt
    # plt.imshow(seg.squeeze())
    # plt.show()

    return seg
