import numpy as np
import sys, os

sys.path.append('/Users/iliailmer/Documents/CUNY/Studies/Fall2018_ComVis/Project/Codes_n_Data/modules')

from EME_Measures import *
from FeatureSimilarity import *
from skimage import exposure

def gamma_correction(image, gamma):
    if gamma > 0:
        return np.power(image, gamma)
    else:
        return np.power(image, 1)


def linear_stretching(I, I_out_min=0, I_out_max=1):
    I_out = I_out_min + (I - I.min()) * (I_out_max - I_out_min) / (I.max() - I.min())
    return I_out


def sigmoid(x):
    return np.divide(1, 1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def contrast_metric(x):
    return (x.max() - x.min()) / (x.max() + x.min())


def double_sigmoid(x, x_1, s):
    return np.sign(x - x_1) * (1 - np.exp(-((x - x_1) / s) ** 2))

def weighted_double_sigmoid(x, x_1, x_2, s_1, s_2, a1, a2):
    return np.sign(x - x_1) * (1 - np.exp(-((x - x_1) / s_1) ** 2))**a1 * (1 - np.exp(-((x - x_2) / s_2) ** 2))**a2


def nonlinear_stretching(I, func="sigmoid", gamma=2, alpha=0.1, beta=0.1, k=1, b=0, x_1=0, s=1):
    I_out_min = 0
    I_out_max = 1
    I_out=0
    if func == "sigmoid":
        I_out = I_out_min + (I - I.min()) / (I.max() - I.min()) * sigmoid(alpha * (I - beta))

    if func == "tanh":
        I_out = I_out_min + (tanh(alpha * (I - beta)) - I.min()) / (I.max() - I.min())  # *tanh(alpha*(I-beta))

    if func == "gcor":
        I_out = I_out_min + (I_out_max - I_out_min) * gamma_correction((I - I.min()) / (I.max() - I.min()), gamma=gamma)

    if func == "dsigmoid":
        a = 1 / (double_sigmoid(k * (1 - b), x_1, s) - double_sigmoid(-k * (1 + b), x_1, s))
        I_out = a * (double_sigmoid(k * (I - b), x_1, s) - double_sigmoid(-k * (I + b), x_1, s))
    if func == "dsigmoid_weighted":
        a = 1 / (weighted_double_sigmoid(k * (1 - b), x_1, s) - weighted_double_sigmoid(-k * (1 + b), x_1, s))
        I_out = a * (weighted_double_sigmoid(k * (I - b), x_1, s) - weighted_double_sigmoid(-k * (I + b), x_1, s))
    return I_out


def negative(image):
    if len(image.shape) > 2:
        negative_im = np.zeros(image.shape)
        for channel in range(image.shape[-1]):
            negative_im[:, :, channel] = image[:, :, channel].max() - image[:, :, channel]
    else:
        negative_im = np.zeros(image.shape)
        negative_im = image.max() - image
    return negative_im


def get_histogram(image):
    pix = image.ravel()  # 1D array of pixels
    L = 256  # len(np.unique(image))  # number of unique pixel intensities

    hist_, bin_edges = np.histogram(pix, bins=L)
    bin_centers = bin_edges[:-1] / 2 + bin_edges[1:] / 2
    return hist_, bin_centers

def get_cdf(image):
    hist_, _ = get_histogram(image)
    CDF = hist_.cumsum() / hist_.cumsum()[-1]
    return CDF


def hist_eq(image, quick=True):
    pix = image.ravel()  # 1D array of pixels
    L = len(np.unique(image))  # number of unique pixel intensities

    hist_, bin_centers = get_histogram(image)
    cdf_ = get_cdf(image)

    if quick:
        hist_equalized = np.interp(image.flatten(), bin_centers, cdf_)
    else:
        hist_equalized = np.zeros(pix.shape)
        for v in range(len(pix)):
            pixel = pix[v]
            s = np.sum(image == pixel)
            idx = hist_.tolist().index(s)
            hist_equalized[v] = np.round((L - 1) * (cdf_[idx] - cdf_.min()) / (len(pix) - cdf_.min()))
    return hist_equalized.reshape(image.shape)


def entropy(probabilities, average=True):
    e = 0
    for proba in probabilities:
        if proba != 0:
            e += -proba * np.log2(proba)
    if average:
        return e#/probabilities.size
    else: return e


def edges(image, method='entropy', win_h=11, win_w=11):
    H = np.int(np.floor(win_h / 2))  # range in height, distance from the center of the window
    W = np.int(np.floor(win_w / 2))  # range in width, same as above
    image_ = np.pad(image, ((W, W - 1), (H, H - 1)), mode='co')
    height, width = image_.shape
    edg = []

    for row in range(0 + H, height - H + 1):
        for column in range(0 + W, width - W + 1):

            window = image_[row - H:row + H + 1, column - W:column + W + 1]

            hist_w, b = np.histogram(window, bins=window.shape[0] * window.shape[1])
            if method == 'entropy':
                ent = entropy(hist_w / (window.shape[0] * window.shape[1]))
                edg.append(ent)
            if method == 'eme':
                eme = EME(window, window_height=win_h, window_width=win_h)
                edg.append(eme)
    return np.asarray(edg)


def get_stacked_histogram(image):
    pix = image.ravel()  # 1D array of pixels
    L = 256  # len(np.unique(image))  # number of unique pixel intensities
    hist_, _ = np.histogram(pix, bins=L)
    hist_ = np.hstack((hist_, mean(image), median(image), std(image), entropy(image)))
    return hist_


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

def rescale(image, min_intensity = 0, max_intensity = 1):
    return exposure.rescale_intensity(image, out_range=(min_intensity,max_intensity))
