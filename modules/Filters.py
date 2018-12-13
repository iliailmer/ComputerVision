import numpy as np
from FeatureSimilarity import median
from scipy.signal import convolve2d


# averaging filter
def averaging(image, win_h=11, win_w=11):
    H = np.int(np.floor(win_h / 2))  # range in height, distance from the center of the window
    W = np.int(np.floor(win_w / 2))  # range in width, same as above
    image_ = np.pad(image, ((W, W - 1), (H, H - 1)), mode='constant')
    height, width = image_.shape
    avg = []

    for row in range(0 + H, height - H + 1):
        for column in range(0 + W, width - W + 1):
            window = image_[row - H:row + H + 1, column - W:column + W + 1]
            avg.append(np.sum(window.flatten()) / window.size)

    return np.asarray(avg).reshape(image.shape)


# alpha-trimmed filter
def alpha_trimmed(image, win_h=11, win_w=11, d=2):
    H = np.int(np.floor(win_h / 2))  # range in height, distance from the center of the window
    W = np.int(np.floor(win_w / 2))  # range in width, same as above
    image_ = np.pad(image, ((W, W - 1), (H, H - 1)), mode='constant')
    height, width = image_.shape
    res = []
    for row in range(0 + H, height - H + 1):
        for column in range(0 + W, width - W + 1):
            window = np.sort(image_[row - H:row + H + 1, column - W:column + W + 1].flatten())
            window = window[d:window.size - d]
            res.append(np.median(window))

    return np.asarray(res).reshape(image.shape)


def median_filter(image, win_h=11, win_w=11):
    H = np.int(np.floor(win_h / 2))  # range in height, distance from the center of the window
    W = np.int(np.floor(win_w / 2))  # range in width, same as above
    image_ = np.pad(image, ((W, W - 1), (H, H - 1)), mode='constant')
    height, width = image_.shape
    res = []

    for row in range(0 + H, height - H + 1):
        for column in range(0 + W, width - W + 1):
            window = image_[row - H:row + H + 1, column - W:column + W + 1]
            res.append(median(window))

    return np.asarray(res).reshape(image.shape)


def gaussian_filter(image=None, size=11, sigma=1., mu=0.):
    rnge = np.arange(np.floor(-size / 2) + 1, np.floor(size / 2) + 1).astype(int)
    [x, y] = np.meshgrid(rnge, rnge)
    f = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2) - (y - mu) ** 2 / (2 * sigma ** 2));
    f = f / np.sum(f)
    W = np.int(np.floor(size/2))
    H = np.int(np.floor(size/2))
    image_ = np.pad(image, ((H, H), (W, W)), mode='constant')
    return convolve2d(image_, f, mode='valid')


def sharpening(image, flt, scale=1, **kwargs):
    return image + scale * (image - flt(image, **kwargs))


# add noise: gaussian, uniform, salt'n'pepper
def add_noise(image, kind='gauss', sigma=0.01, par=100):
    if kind == 'gauss':
        n = np.random.normal(loc=0, scale=sigma, size=image.size).reshape(image.shape)
        return image + n
    elif kind == 'saltpepper':
        noise = np.random.randint(par, size=(image.shape[0], image.shape[1]))
        image = np.where(noise == 0, 0, image)
        image = np.where(noise == (par - 1), image.max(), image)
        return image
    elif kind == 'uniform':
        noise = np.random.randint(par, size=(image.shape[0], image.shape[1]))
        image = np.where(noise == (par - 1), 1, image)
        return image


# adaptive local noise reduction filter
def ALNR_Filter(image, win_h=3, win_w=3):
    H = np.int(np.floor(win_h / 2))  # range in height, distance from the center of the window
    W = np.int(np.floor(win_w / 2))  # range in width, same as above
    image_ = np.pad(image, ((W, W - 1), (H, H - 1)), mode='constant')
    height, width = image_.shape
    local_var = []
    local_mean = []
    for row in range(0 + H, height - H + 1, 1):
        for column in range(0 + W, width - W + 1, 1):
            window = image_[row - H:row + H + 1, column - W:column + W + 1]
            local_var.append(np.var(window))
            local_mean.append(np.mean(window))
    SHAPE = (np.int(np.sqrt(len(local_var))), np.int(np.sqrt(len(local_var))))
    local_var = np.asarray(local_var).reshape(SHAPE)
    SHAPE = (np.int(np.sqrt(len(local_mean))), np.int(np.sqrt(len(local_mean))))
    local_mean = np.asarray(local_mean).reshape(SHAPE)
    noise_var = np.var(local_var)
    local_var = np.where(local_var < noise_var, noise_var, local_var)

    return image - noise_var / local_var * (image - local_mean)


def hp_filter(image):
    high_pass = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])
    image_ = np.pad(image, ((2, 2), (2, 2)), mode='constant')
    return convolve2d(image_, high_pass, mode='valid')


def median_filter_weighted(image, win_h=3, win_w=3):
    weight = np.asarray([1,2,4,1,2,3,3,6,3])
    H = np.int(np.floor(win_h / 2))  # range in height, distance from the center of the window
    W = np.int(np.floor(win_w / 2))  # range in width, same as above
    image_ = np.pad(image, ((W, W-1), (H, H-1)), mode='constant')
    height, width = image_.shape
    res = []
    for row in range(0 + H, height - H + 1):
        for column in range(0 + W, width - W + 1):
            window = image_[row - H:row + H + 1, column - W:column + W + 1].reshape((-1,1)).tolist()
            m = np.hstack([a*b for a,b in zip(weight,window)])
            res.append(np.median(m))

    return np.asarray(res).reshape(image.shape)