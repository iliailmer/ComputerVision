import numpy as np


def EME(image, window_height=11, window_width=11):
    """
    EME measure for showing image quality based on human visual system.
    For details, see
    Agaian, Sos S., Karen Panetta,
    and Artyom M. Grigoryan.
    "A new measure of image enhancement."
    IASTED International Conference on Signal Processing & Communication.
    Citeseer, 2000.

    :param image: input image, must be single-channel.
    :param window_height: height of the inspecting window
    :param window_width: width of the inspecting window
    :return: A real-valued enhancement measure.
    """
    height, width = image.shape
    # k_1 = np.floor(height/window_height)
    # k_2 = np.floor(width/window_width)
    sum_ = 0
    k = 0
    H = np.int(np.floor(window_height / 2))  # range in height, distance from the center of the window
    W = np.int(np.floor(window_width / 2))  # range in width, same as above
    for row in range(0 + H, height - H + 1, window_height):
        for column in range(0 + W, width - W + 1, window_width):

            window = image[row - H:row + H + 1, column - W:column + W + 1]

            I_max = window.max()
            I_min = window.min()

            D = (I_max + 1) / (I_min + 1)
            if D < 0.02:
                D = 0.02
            k += 1
            sum_ += 20 * np.log(D)
        # sum_k_1 += sum_k_2
        # sum_k_2 = 0

    eme = sum_ / k
    return eme


def EME_color(image):
    """
    Application of EME to color images.
    :param image: color image (multi-channel)
    :return: EME of that image
    """
    emes_ = []
    if image.shape[-1]>1:
        for each in range(image.shape[-1]):
            emes_.append(EME(image[:,:,each]))
    else:
        emes_.append(EME(image))
    return max(emes_)

