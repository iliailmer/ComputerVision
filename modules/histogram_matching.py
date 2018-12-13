from ColorTransforms import *
from DataLoader import *
from EME_Measures import *
from Enhancement import *
from Plotter import *
from Thresholding import *

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt


def hist_match(source, template):
    s_shape = source.shape  # source shape
    source = source.ravel()
    template = template.ravel()

    source_uniques, source_bins, source_counts =np.unique(source, return_inverse=True, return_counts=True)
    template_uniques, template_counts = np.unique(template, return_counts=True)

    source_quantiles = np.cumsum(source_counts).astype(np.float64)
    source_quantiles /=source_quantiles[-1]

    template_quantiles = np.cumsum(template_counts).astype(np.float64)
    template_quantiles /=template_quantiles[-1]

    interp_t_values = np.interp(source_quantiles,
                                template_quantiles,
                                template_uniques)

    return interp_t_values[source_bins].reshape(s_shape)


image_healthy = np.load('data/image_1.npy')
image_pneum = np.load('data/image_2.npy')

import scipy.misc as m
from skimage.data import camera
plt.set_cmap('gray')

ascent = m.ascent()


plt.imshow(hist_match(camera(), ascent))
plt.show()

