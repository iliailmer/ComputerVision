from skimage.transform import resize
import skimage.filters as filters
from skimage.exposure import equalize_hist, histogram
from skimage import color
from skimage.measure import shannon_entropy
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import pyramid_gaussian
from skimage.transform import pyramid_laplacian
from skimage.transform import pyramid_expand
from skimage.transform import pyramid_reduce
from skimage.feature import hog

from sklearn.feature_extraction.image import extract_patches_2d
import pywt

from scipy.ndimage import convolve, binary_erosion, generate_binary_structure

import matplotlib.pyplot as plt

from ColorTransforms import *
from EME_Measures import *
from Enhancement import *
from FeatureSimilarity import *
from Filters import *
from Plotter import *
from Thresholding import *
from DataLoader import *

def plot_histogram_comparison(image1,image2):
    """
    Plots two histograms of two images side-by-side for visual comparison
    :param image1: First image
    :param image2: Second image
    :return: plots of two histograms
    """
    hist1,  bins1  = get_histogram(image1)
    hist2,  bins2 = get_histogram(image2)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,8))

    ax[0].bar(bins1, hist1/image1.size, align='center', width = 0.7 * (bins1[1] - bins1[0]))
    ax[1].bar(bins2, hist2/image2.size, align='center', width = 0.7 * (bins2[1] - bins2[0]))
    

def plot_cdf_comparison(image1,image2):
    """
    PLots two Cumulative Distribution Functions for two images
    :param image1: first image
    :param image2: second image
    :return: plot
    """
    cdf1  = get_cdf(image1)
    cdf2  = get_cdf(image2)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,8))

    ax[0].plot(cdf1)
    ax[1].plot(cdf2)


def moment_n(dist, bins, n):
    """
    Calculates n-th moment of a distribution

    :param dist: the distribution vector
    :param bins: number of bins in the ditribution (how many columns in a histogram)
    :param n: order of the moment
    :return: real-valued moment
    """
    return np.sum(np.power((bins),n)*dist)


def central_moment_n(dist, bins, n):
    """
        Calculates CENTERED n-th moment of a distribution

        :param dist: the distribution vector
        :param bins: number of bins in the ditribution (how many columns in a histogram)
        :param n: order of the moment
        :return: real-valued moment
        """
    return np.sum(np.power((bins - moment_n(dist,bins, 1)),n)*dist)


def get_first_order_statistics(image):
    """
    Returns first-order statistics with respect to an input image.
    Calculates histogram (probability distribution)
    :param image: input image
    :return: image mean, standard deviation, smoothness, skewness, kurtosis, and entropy
    """
    hist_, bins_ = get_histogram(image)
    hist_ = hist_/image.size

    mean_ = moment_n(hist_,bins_, 1)
    variance = central_moment_n(hist_,bins_, 2)
    std = np.sqrt(variance)
    smoothness = 1-1/(1+variance)
    skewness = central_moment_n(hist_,bins_, 3)/np.power(variance, 3/2)
    kurtosis = central_moment_n(hist_,bins_, 4)/np.power(variance, 4/2)
    entr_ = entropy(probabilities=hist_)
    return np.asarray([mean_, std, smoothness, skewness, kurtosis, entr_])


def show_patches(patches):
    """
    I used this helper-function to plot two patches of an image side-by-side for visual comparison
    :param patches: list of patches (numpy arrays)
    :return: plots
    """

    fig, ax = plt.subplots(nrows=1,ncols = len(patches), figsize=(6,6));
    if len(patches)>1:
        for i in range(len(patches)):
            ax[i].imshow(patches[i]);
    elif len(patches)==1:
        ax.imshow(patches[0]);
    else:
        print('No object found')


def squarify(image, val):
    """
    Pad a given image with value val to make it square
    :param image: image
    :param val: value to pad with
    :return:
    """
    (a,b)=image.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(image, padding, mode='constant', constant_values=val)


def plot_feature_comparison_g(images_enhanced, indices_h, indices_p, level = 0, npat = 0,  ):
    """
    plots two stem-graphs of Haralik features for images.
    
    level: number of a level in gaussian pyramid
    """
    #index = np.random.randint(0,100)
    inx = npat # number of patient
    # HARALICK FEATURES FOR HEALTHY IMAGE
    enh =images_enhanced[indices_h[inx],...] #this is probably inefficient but I had to do it fast!

    P =tuple(pyramid_gaussian(enh,multichannel=False))
    ind = level #number of pyramid layer

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    har_feat = np.zeros((3*4*6))

    glcm = greycomatrix(P[ind].astype('uint8'), distances, angles, 
                            256, symmetric=True, normed=True)    
    feats = np.hstack([std_sc(greycoprops(glcm, prop).ravel()) for prop in properties])
    har_feat = np.vstack([har_feat, feats])#, std_sc(haralick_entropy)])])

    har_feat = har_feat[1:]

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20,20))
    markerline, stemlines, baseline = ax[0,0].stem(range(har_feat.size), har_feat[0])
    ax[1,0].imshow(P[ind])

    for i in range(har_feat.size):
        stemlines[i].set_color('g')

    ax[1,0].set_title(f"Class:{target[indices_h[inx]]}");

    # HARALICK FEATURES FOR PNEUMONIA IMAGE GLOBALLY

    enh =images_enhanced[indices_p[inx],...]

    P =tuple(pyramid_gaussian(enh,multichannel=False))
    ind = level

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    har_feat_ = np.zeros((3*4*6))

    glcm = greycomatrix(P[ind].astype('uint8'), distances, angles, 
                            256, symmetric=True, normed=True)

    feats = np.hstack([std_sc(greycoprops(glcm, prop).ravel()) for prop in properties])
    har_feat_ = np.vstack([har_feat_, feats])#np.hstack([])])#, std_sc(haralick_entropy)])])

    har_feat_ = har_feat_[1:]

    markerline, stemlines, baseline = ax[0,1].stem(range(har_feat_.size), har_feat_[0])
    ax[1,1].imshow(P[ind])
    ax[1,1].set_title(f"Class:{target[indices_p[inx]]}");
    for i in range(har_feat.size):
        stemlines[i].set_color('r')

    patient =  patients[patients["id"] == ids_[indices_p[inx]]]
    
    ax[0,0].set_ylim([0,np.maximum(har_feat_.max(), har_feat.max())]);
    ax[0,1].set_ylim([0,np.maximum(har_feat_.max(), har_feat.max())]);
    

def plot_feature_comparison_l(level = 0, npat = 0, prop = 'contrast'):
    """
    plots two stem-graphs of Haralik features for images.
    
    level: number of a level in gaussian pyramid
    """
    indices_p = list(np.where(target==1))[0]
    indices_h = list(np.where(target==0))[0]
    inx = npat # number of patient
    enh =images_enhanced[indices_h[inx],...]

    P =tuple(pyramid_laplacian(enh,multichannel=False))
    ind = level #number of pyramid layer

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']#'contrast', 
    har_feat = np.zeros((3*4*6))

    glcm = greycomatrix(P[ind].astype('uint8'), distances, angles, 
                            256, symmetric=True, normed=True)
    haralick_entropy = np.zeros((12))
    k=0
    for p in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            haralick_entropy[k] = shannon_entropy(glcm[...,p,j])
            k+=1
    feats = np.hstack([std_sc(greycoprops(glcm, prop).ravel()) for prop in properties])
    har_feat = np.vstack([har_feat, feats])#np.hstack([, haralick_entropy])])

    har_feat = har_feat[1:]
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20,20))
    markerline, stemlines, baseline = ax[0,0].stem(range(har_feat.size), har_feat[0])
    ax[1,0].imshow(P[ind])

    for i in range(har_feat.size):
        stemlines[i].set_color('g')

    ax[1,0].set_title(f"Class:{target[indices_h[inx]]}");

    enh =images_enhanced[indices_p[inx],...]

    P =tuple(pyramid_laplacian(enh,multichannel=False))
    ind = level

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']#
    har_feat_ = np.zeros((3*4*6))

    glcm = greycomatrix(P[ind].astype('uint8'), distances, angles, 
                            256, symmetric=True, normed=True)

    feats = np.hstack([std_sc(greycoprops(glcm, prop).ravel()) for prop in properties])
    har_feat_ = np.vstack([har_feat_, feats])#np.hstack([, haralick_entropy])])#, haralick_entropy

    har_feat_ = har_feat_[1:]
    markerline, stemlines, baseline = ax[0,1].stem(range(har_feat_.size), har_feat_[0])
    ax[1,1].imshow(P[ind])
    ax[1,1].set_title(f"Class:{target[indices_p[inx]]}");
    for i in range(har_feat.size):
        stemlines[i].set_color('r')
    patient =  patients[patients["id"] == ids_[indices_p[inx]]] 
    ax[0,0].set_ylim([0,np.maximum(har_feat_.max(), har_feat.max())]);
    ax[0,1].set_ylim([0,np.maximum(har_feat_.max(), har_feat.max())]);

def plot_feature_comparison_alpha(level = 0, npat = 0, prop = 'contrast'):
    """
    plots two stem-graphs of Haralik features for images.
    
    level: number of a level in gaussian pyramid
    """
    indices_p = list(np.where(target==1))[0]
    indices_h = list(np.where(target==0))[0]
    #index = np.random.randint(0,100)
    inx = npat # number of patient
    # HARALICK FEATURES FOR HEALTHY IMAGE (GLOBALLY)
    enh =images_enhanced[indices_h[inx],...]

    P =alpha_pyramid_reduce(d=5,downscale=2,image=enh, max_level=level, win_h=5,win_w=5)
    ind = level #number of pyramid layer

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']#'contrast', 
    har_feat = np.zeros((3*4*6))

    glcm = greycomatrix(P[ind].astype('uint8'), distances, angles, 
                            256, symmetric=True, normed=True)
    haralick_entropy = np.zeros((12))
    k=0
    for p in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            haralick_entropy[k] = shannon_entropy(glcm[...,p,j])
            k+=1
    feats = np.hstack([std_sc(greycoprops(glcm, prop).ravel()) for prop in properties])
    har_feat = np.vstack([har_feat, feats])#np.hstack([, haralick_entropy])])

    har_feat = har_feat[1:]#(har_feat[1:]-np.min(har_feat[1:]))/(np.max(har_feat[1:])-np.min(har_feat[1:]))
                                                           #/np.sum(har_feat_[1:])

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20,20))
    markerline, stemlines, baseline = ax[0,0].stem(range(har_feat.size), har_feat[0])
    ax[1,0].imshow(P[ind])

    for i in range(har_feat.size):
        stemlines[i].set_color('g')

    ax[1,0].set_title(f"Class:{target[indices_h[inx]]}");

    enh =images_enhanced[indices_p[inx],...]

    P = alpha_pyramid_reduce(d=5,downscale=2,image=enh, max_level=level, win_h=5,win_w=5)
    ind = level

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']#
    har_feat_ = np.zeros((3*4*6))

    glcm = greycomatrix(P[ind].astype('uint8'), distances, angles, 
                            256, symmetric=True, normed=True)

    feats = np.hstack([std_sc(greycoprops(glcm, prop).ravel()) for prop in properties])
    har_feat_ = np.vstack([har_feat_, feats])
    har_feat_ = har_feat_[1:]
    markerline, stemlines, baseline = ax[0,1].stem(range(har_feat_.size), har_feat_[0])
    ax[1,1].imshow(P[ind])
    ax[1,1].set_title(f"Class:{target[indices_p[inx]]}");
    for i in range(har_feat.size):
        stemlines[i].set_color('r')

    patient =  patients[patients["id"] == ids_[indices_p[inx]]] 
    ax[0,0].set_ylim([0,np.maximum(har_feat_.max(), har_feat.max())]);
    ax[0,1].set_ylim([0,np.maximum(har_feat_.max(), har_feat.max())]);


def get_patches(index, draw=True):
    indices_p = list(np.where(target==1))[0]
    indices_h = list(np.where(target==0))[0]
    indices_p = indices_p[:min(len(indices_h),len(indices_p))]
    indices_h = indices_h[:min(len(indices_h),len(indices_p))]
    
    image_healthy = images_enhanced[indices_h[index]]
    image_pneumonia = images_enhanced[indices_p[index]]
    
    good_patches = []
    bad_patches = []
    patient =  patients[patients["id"] == ids_[indices_p[index]]]
    for i in range(len(patient)):
        x = int(patient.iloc[i].x/2)
        y = int(patient.iloc[i].y/2)
        w = int(patient.iloc[i].width/2)
        h = int(patient.iloc[i].height/2)
        good_patches.append(np.asarray(image_healthy[y:y+h, x:x+w,].ravel().tolist(), 
                                       dtype="uint8").reshape((h,w)))
        bad_patches.append(np.asarray(image_pneumonia[y:y+h, x:x+w,].ravel().tolist(), 
                                       dtype="uint8").reshape((h,w)))
    if draw:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize= (12,8))
        ax[0].imshow(image_healthy)
        ax[0].set_title("Healthy")
        ax[1].imshow(image_pneumonia)
        ax[1].set_title("Pneumonic")
        for i in range(len(patient)):
            x = int(patient.iloc[i].x/2)
            y = int(patient.iloc[i].y/2)
            w = int(patient.iloc[i].width/2)
            h = int(patient.iloc[i].height/2)
            ax[0].plot([x, x + w], [y, y], color='r')
            ax[0].plot([x + w, x + w], [y, y + h], color='r')
            ax[0].plot([x, x], [y, y + h], color='r')
            ax[0].plot([x, x + w], [y + h, y + h], color='r')
            ax[1].plot([x, x + w], [y, y], color='r')
            ax[1].plot([x + w, x + w], [y, y + h], color='r')
            ax[1].plot([x, x], [y, y + h], color='r')
            ax[1].plot([x, x + w], [y + h, y + h], color='r')
    return good_patches, bad_patches


def plot_patch_feature_comparison(patch_h, patch_p, level = 0, npat = 0, mode = 'gauss'):
    """
    plots two stem-graphs of Haralik features for images.
    
    level: number of a level in gaussian pyramid
    """
    indices_p = list(np.where(target==1))[0]
    indices_h = list(np.where(target==0))[0]
    indices_p = indices_p[:min(len(indices_h),len(indices_p))]
    indices_h = indices_h[:min(len(indices_h),len(indices_p))]

    #index = np.random.randint(0,100)
    inx = npat # number of patient
    # HARALICK FEATURES FOR HEALTHY IMAGE (GLOBALLY)
    enh = patch_h
    if mode=='gauss':
        P =tuple(pyramid_gaussian(enh, multichannel=False))
    elif mode=='laplacian':
        P =tuple(pyramid_laplacian(enh, multichannel=False))
    else:
        P = alpha_pyramid_reduce(d=5,downscale=2,image=enh, max_level=level, win_h=5,win_w=5)
    ind = level #level of pyramid layer

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    har_feat = np.zeros((3*4*6))

    glcm = greycomatrix(rescale(P[ind],255).astype('uint8'), distances, angles, 
                            256, symmetric=True, normed=True)
    haralick_entropy = np.zeros((12))
    """k=0
    for p in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            haralick_entropy[k] = shannon_entropy(glcm[...,p,j])
            k+=1"""
    feats = np.hstack([std_sc(greycoprops(glcm, prop).ravel()) for prop in properties])
    har_feat = np.vstack([har_feat, feats])#np.hstack([, haralick_entropy])])

    har_feat = har_feat[1:]

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20,20))
    markerline, stemlines, baseline = ax[0,0].stem(range(har_feat.size), har_feat[0])
    ax[1,0].imshow(P[ind])

    for i in range(har_feat.size):
        stemlines[i].set_color('g')

    ax[1,0].set_title(f"Class:{target[indices_h[inx]]}");

    enh = patch_p
    if mode=='gauss':
        P =tuple(pyramid_gaussian(enh, multichannel=False))
    elif mode=='laplacian':
        P =tuple(pyramid_laplacian(enh, multichannel=False))
    else:
        P = alpha_pyramid_reduce(d=5,downscale=2,image=enh, max_level=level, win_h=5,win_w=5)
    
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    har_feat_ = np.zeros((3*4*6))

    glcm = greycomatrix(rescale(P[ind],255).astype('uint8'), distances, angles, 
                            256, symmetric=True, normed=True)
    """haralick_entropy = np.zeros((12))
    k=0
    for p in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            haralick_entropy[k] = shannon_entropy(glcm[...,p,j])
            k+=1
    """
    feats = np.hstack([std_sc(greycoprops(glcm, prop).ravel()) for prop in properties])
    har_feat_ = np.vstack([har_feat_, feats])#np.hstack([feats, haralick_entropy])])

    har_feat_ = har_feat_[1:]

    markerline, stemlines, baseline = ax[0,1].stem(range(har_feat_.size), har_feat_[0])
    ax[1,1].imshow(P[ind])
    ax[1,1].set_title(f"Class:{target[indices_p[inx]]}");
    for i in range(har_feat.size):
        stemlines[i].set_color('r')  
    ax[0,1].set_ylim([0,np.maximum(har_feat_.max(), har_feat.max())]);
    ax[0,0].set_ylim([0,np.maximum(har_feat_.max(), har_feat.max())]);


def collect_features(image):
    #total will be: 7 haralick features, 256
    if image.mean()>120:
        image = image.max()-image
    distances = [1, 2, 3]
    levels = list(range(3))
    n_points=8
    radius=1
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    all_feat = np.array([])#np.zeros((3*4*6+12+256))
    #collect gaussian pyramids
    p_ = tuple(pyramid_gaussian(image=image,multichannel=False,max_layer=levels[-1]+1))
    for each_level in levels:
        glcm = greycomatrix(rescale(p_[each_level],255).astype('uint8'), distances, angles, 
                            256, symmetric=True, normed=True)
        '''haralick_entropy = np.zeros((12))
        k=0
        for p in range(glcm.shape[2]):
            for j in range(glcm.shape[3]):
                haralick_entropy[k] = shannon_entropy(glcm[...,p,j])
                k+=1'''
        feats = np.hstack([std_sc(greycoprops(glcm, prop).ravel()) for prop in properties])
        all_feat = np.concatenate([all_feat,feats])#np.hstack([, haralick_entropy])]) #7*12 haralick features
        lbp = local_binary_pattern(p_[each_level],8,1)
        lbp_hist, _ = np.histogram(lbp, density=True)
        all_feat = np.concatenate([all_feat,std_sc(lbp_hist)])
        icoeffs2 = pywt.dwt2(p_[each_level], wavelet=DaubAgaianWavelet)
        LL, (LH, HL, HH) = coeffs2
        LH_hist, _ = np.histogram(LH, density=True)#get_histogram(LH)
        HH_hist, _ = np.histogram(HH, density=True)#get_histogram(HH)
        HL_hist, _ = np.histogram(HL, density=True)#get_histogram(HL)
        
        all_feat = np.concatenate([all_feat, 
                                   np.hstack([std_sc(LH_hist), std_sc(HH_hist), std(HL_hist)])]) 
    p_ = tuple(pyramid_laplacian(image=image,multichannel=False,max_layer=levels[-1]+1))
    for each_level in levels:
        glcm = greycomatrix(rescale(p_[each_level],255).astype('uint8'), distances, angles, 
                            256, symmetric=True, normed=True)
        feats = np.hstack([std_sc(greycoprops(glcm, prop).ravel()) for prop in properties])
        
        all_feat = np.concatenate([all_feat, feats])#np.hstack([, haralick_entropy])]) #7*12 haralick features
        lbp = local_binary_pattern(p_[each_level],8,1)
        lbp_hist, _ = np.histogram(lbp, density=True)
        all_feat = np.concatenate([all_feat,std_sc(lbp_hist)])
        icoeffs2 = pywt.dwt2(p_[each_level], wavelet=DaubAgaianWavelet)
        LL, (LH, HL, HH) = coeffs2
        LH_hist, _ = np.histogram(LH, density=True)#get_histogram(LH)
        HH_hist, _ = np.histogram(HH, density=True)#get_histogram(HH)
        HL_hist, _ = np.histogram(HL, density=True)#get_histogram(HL)
        
        all_feat = np.concatenate([all_feat, 
                                   np.hstack([std_sc(LH_hist), std_sc(HH_hist), std(HL_hist)])]) 
    p_ = alpha_pyramid_reduce(image=image,d=5,win_h=5,win_w=5,downscale=2,max_level=levels[-1]+1)
    for each_level in levels:
        glcm = greycomatrix(rescale(p_[each_level],255).astype('uint8'), distances, angles, 
                            256, symmetric=True, normed=True)
        feats = np.hstack([std_sc(greycoprops(glcm, prop).ravel()) for prop in properties])
        all_feat = np.concatenate([all_feat,feats])#np.hstack([, haralick_entropy])]) #7*12 haralick features
        lbp = local_binary_pattern(p_[each_level],8,1)
        lbp_hist, _ = np.histogram(lbp, density=True)
        all_feat = np.concatenate([all_feat,std_sc(lbp_hist)])
        icoeffs2 = pywt.dwt2(p_[each_level], wavelet=DaubAgaianWavelet)
        LL, (LH, HL, HH) = coeffs2
        LH_hist, _ = np.histogram(LH, density=True)#get_histogram(LH)
        HH_hist, _ = np.histogram(HH, density=True)#get_histogram(HH)
        HL_hist, _ = np.histogram(HL, density=True)#get_histogram(HL)
        
        all_feat = np.concatenate([all_feat, 
                                   np.hstack([std_sc(LH_hist), std_sc(HH_hist), std(HL_hist)])]) 
    all_feat = all_feat[1:]
    return all_feat

def FibonacciDaubechies(N):
    c0 = (1+np.sqrt(N))/(N+1)
    c1 = (N+np.sqrt(N))/(N+1)
    c2 = (N-np.sqrt(N))/(N+1)
    c3 = (1-np.sqrt(N))/(N+1)
    return [[c0, c1, c2, c3], [c3, -c2, c1, -c0], [c3, c2, c1, c0], [-c0, c1, -c2, c3]]


def BuildBank(N):
    return FibonacciDaubechies(N)


def alpha_pyramid_reduce(image, d, win_h, win_w, downscale = 2, max_level = 4):
    exception = BaseException("Maximal depth level too big!")
    if max_level>np.ceil(np.log2(image.shape[0])):
        raise exception
    #image = squarify(image,median(image))
    result = []
    downscaled = image
    result.append(downscaled)
    for level in range(max_level):
        smoothed = alpha_trimmed(image=downscaled, d=d, win_h=win_h, win_w=win_w)
        downscaled = downscale_local_mean(smoothed, factors=(downscale,downscale))
        result.append(downscaled)
    return result

def std_sc(feats):
    """
    Standard scaler for features
    """
    return (feats-mean(feats))/std(feats)