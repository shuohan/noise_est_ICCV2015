#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-07 14:36:55


import numpy as np
from cv2 import imread
from utils import im2patch, im2double, im2patch_mask, calc_fg_mask
import time
import matplotlib.pyplot as plt


def noise_estimate(im, fg=False, pch_size=8):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, H x W x 3 or H x W numpy tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''

    if im.ndim == 3:
        im = im.transpose((2, 0, 1))
    else:
        im = np.expand_dims(im, axis=0)

    # image to patch
    if fg:
        mask = calc_fg_mask(im)
        pch = im2patch_mask(im, mask, pch_size, 3)
    else:
        pch = im2patch(im, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
    d = pch.shape[0]

    mu = pch.mean(axis=1, keepdims=True)  # d x 1
    X = pch - mu
    sigma_X = np.matmul(X, X.transpose()) / num_pch
    sig_value, _ = np.linalg.eigh(sigma_X)
    sig_value.sort()

    for ii in range(-1, -d-1, -1):
        tau = np.mean(sig_value[:ii])
        if np.sum(sig_value[:ii]>tau) == np.sum(sig_value[:ii] < tau):
            return np.sqrt(tau)


if __name__ == '__main__':
    im = imread('./lena.png')
    im = im2double(im)
    print(np.min(im), np.max(im))

    noise_level = [5, 15, 20, 30, 40]

    def rgb2gray(rgb):
        return np.dot(rgb[..., :], [0.299, 0.587, 0.144])

    im = rgb2gray(im)
    print(im.shape)

    for i, level in enumerate(noise_level):
        sigma = level / 255
        im_noise = im + np.random.randn(*im.shape) * sigma

        start = time.time()
        est_level = noise_estimate(im_noise, 8)
        end = time.time()
        time_elapsed = end -start

        plt.subplot(2, 3, i + 1)
        plt.imshow(im_noise, cmap='gray')
        plt.title('Truth noise level %.2f; estimated %.2f' % (level, est_level*255))

        str_p = "Time: {0:.4f}, Ture Level: {1:6.4f}, Estimated Level: {2:6.4f}"
        print(str_p.format(time_elapsed, level, est_level*255))
    plt.show()
