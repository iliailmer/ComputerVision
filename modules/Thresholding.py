import numpy as np
from Enhancement import get_histogram, entropy
import sys, os

sys.path.append('/Users/iliailmer/Documents/Studies_CUNY/Fall2018_ComVis/Project/Codes_n_Data/modules')

from EME_Measures import EME


def basic_global_thr(image):
    T = np.min(image) / 2 + np.max(image) / 2
    done = False
    while not done:
        g = image > T
        T_new = np.mean(image[np.where(image > T)]) / 2 + np.mean(image[np.where(image <= T)]) / 2
        done = np.abs(T - T_new) < 0.5
        T = T_new
    return T


def spatially_adaptive_thresholding(image, win_h=11, win_w=11, kind='EME'):
    H = np.int(np.floor(win_h / 2))  # range in height, distance from the center of the window
    W = np.int(np.floor(win_w / 2))  # range in width, same as above
    height, width = image.shape
    edg = []
    res = np.zeros_like(image)
    for row in range(0 + H, height - H + 1, 1):
        for column in range(0 + W, width - W + 1, 1):

            window = image[row - H:row + H + 1, column - W:column + W + 1]
            if kind == 'EME':
                res[row - H:row + H + 1, column - W:column + W + 1] = window > EME(window)/ window.size
            elif kind == 'entropy':
                hist, _ = get_histogram(window)
                res[row - H:row + H + 1, column - W:column + W + 1] = window > entropy(hist / window.size)/ window.size
    return res
