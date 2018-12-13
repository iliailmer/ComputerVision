import numpy as np
import scipy as sp
from scipy import constants

from skimage import color

"""
This file contains some color transformations for images
"""

def recoloring(image):
    """
    Colors a greyscale image using golden ratio
    :param image: a grey-level original image
    :return: a resulting RGB image, same dimensions plus 3 channels
    """
    shape = image.shape
    colored_image_rgb = np.zeros((shape[0], shape[1], 3))

    colored_image_rgb[:, :, 0] = constants.golden*image
    colored_image_rgb[:, :, 1] = np.sqrt(constants.golden)*image
    colored_image_rgb[:, :, 2] = image
    
    return (colored_image_rgb)

def to_cmy(image):
    """
    Convert RGB to CMY
    :param image: original. If grey-level then function returns negative of it
    :return: CMY image or negative of original
    """
    return np.double(1-image/image.max())


def to_cmyk(image):
    """
    Conversion to CMYK from original

    :param image: original
    :return: CMYK image
    """
    cmy = to_cmy(image)
    image_cmyk = np.zeros((image.shape[0],image.shape[1],4))
    min_ = np.min(cmy, axis=-1)
    if not min_.all():
        image_cmyk[...,0] = np.divide(cmy[...,0]-min_, 1-min_, 
                                      out=np.zeros_like(cmy[...,0]-min_), 
                                      where=(1-min_)!=0)
        image_cmyk[...,1] = np.divide(cmy[...,1]-min_, 1-min_, 
                                      out=np.zeros_like(cmy[...,0]-min_), 
                                      where=(1-min_)!=0)
        image_cmyk[...,2] = np.divide(cmy[...,2]-min_, 1-min_, 
                                      out=np.zeros_like(cmy[...,0]-min_), 
                                      where=(1-min_)!=0)
        image_cmyk[...,3] = min_
    return np.uint8(image_cmyk*100)


def to_hsi(image_rgb):
    """
    Conversion to Hue, Saturation, intensity from original

    :param image: original, 3-channel image.
    :return: CMYK image
    """
    image_rgb = image_rgb.astype("float32")
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]
    
    I = (R + G + B) / 3
    
    R = R
    G = G
    B = B
    numerator = 0.5 * (R - G + R - B)
    denominator = np.sqrt((np.abs(R - G)) ** 2 + (R - B) * (G - B))
    theta = np.arccos(np.divide(numerator, denominator,
                                out=np.ones_like(numerator),
                                where=denominator>numerator))

    if (B <= G).all():
        H = np.rad2deg(theta)
    else:
        H = 360 - np.rad2deg(theta)
    
    S = 1 - 3 * np.divide(np.min(image_rgb,axis=-1),(R + G + B),
                          out = np.zeros_like(R),
                          where = (R + G + B)!=0)

    image_hsi = np.zeros_like(image_rgb)

    image_hsi[:, :, 0] = H
    image_hsi[:, :, 1] = S
    image_hsi[:, :, 2] = I

    return (image_hsi)


def to_xyz(image_rgb):
    """
    Conversion to XYZ from RGB (or any 3 channel image)

    :param image_rgb: original image
    :return: XYZ image
    """
    image_rgb = image_rgb.astype("float")
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]

    X = (0.49 * R + 0.31 * G + 0.20 * B) / 0.17
    Y = (0.17 * R + 0.81 * G + 0.01 * B) / 0.17
    Z = (0.00 * R + 0.01 * G + 0.99 * B) / 0.17

    image_xyz = np.zeros_like(image_rgb)
    image_xyz[:, :, 0] = X
    image_xyz[:, :, 1] = Y
    image_xyz[:, :, 2] = Z

    return image_xyz


def xyz2rgb(image_xyz):
    """
    Inverse transformation from XYZ to RGB
    :param image_xyz: XYZ original
    :return: RGB result
    """
    X = image_xyz[:, :, 0]
    Y = image_xyz[:, :, 1]
    Z = image_xyz[:, :, 2]

    image_rgb = np.zeros_like(image_xyz)
    image_rgb[:, :, 0] = 3.24079 * X - 1.537150 * Y - 0.498535 * Z
    image_rgb[:, :, 1] = -0.969256 * X + 1.875992 * Y + 0.041556 * Z
    image_rgb[:, :, 1] = 0.055648 * X - 0.204043 * Y + 1.057311 * Z

    return (image_rgb)


def to_yiq(image_rgb):
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]

    image_yiq = np.zeros_like(image_rgb)
    image_yiq[:, :, 0] = 0.299 * R + 0.587 * G + 0.114 * B
    image_yiq[:, :, 1] = 0.596 * R - 0.275 * G - 0.312 * B
    image_yiq[:, :, 2] = 0.212 * R - 0.528 * G + 0.311 * B

    return image_yiq


def to_yuv(image_rgb):
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]

    image_yuv = np.zeros_like(image_rgb)
    image_yuv[:, :, 0] = 0.299 * R + 0.587 * G + 0.114 * B
    image_yuv[:, :, 1] = 0.492 * (B - image_yuv[:, :, 0])
    image_yuv[:, :, 2] = 0.866 * (R - image_yuv[:, :, 0])

    return image_yuv


def to_ycbcr(image_rgb):
    R = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    B = image_rgb[:, :, 2]

    image_ycbcr = np.zeros_like(image_rgb)
    image_ycbcr[:, :, 0] = 0.299 * R + 0.587 * G + 0.114 * B
    image_ycbcr[:, :, 1] = 0.772 * (B - image_ycbcr[:, :, 0]) + 0.5
    image_ycbcr[:, :, 2] = 0.402 * (R - image_ycbcr[:, :, 0]) + 0.5

    return image_ycbcr
