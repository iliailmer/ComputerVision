from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, precision_score, roc_curve, \
    precision_recall_curve
from sklearn.model_selection import train_test_split, KFold

from modules.EME_Measures import EME
from modules.Filters import gaussian_filter, alpha_trimmed
from modules.Enhancement import nonlinear_stretching
from modules.features import std_sc
import sys
import pywt
import os
from skimage import filters, exposure
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.exposure import rescale_intensity
from skimage.transform import downscale_local_mean, pyramid_gaussian, pyramid_laplacian
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops

plt.style.use('default')
plt.set_cmap("gray")


def rescale(image, min_intensity: object = 0, max_intensity: object = 1) -> object:
    return rescale_intensity(image, out_range = (min_intensity, max_intensity))


def FibonacciDaubechies(N):
    c0 = (1 + np.sqrt(N)) / (N + 1)
    c1 = (N + np.sqrt(N)) / (N + 1)
    c2 = (N - np.sqrt(N)) / (N + 1)
    c3 = (1 - np.sqrt(N)) / (N + 1)
    return [[c0, c1, c2, c3], [c3, -c2, c1, -c0], [c3, c2, c1, c0], [-c0, c1, -c2, c3]]


def BuildBank(N):
    return FibonacciDaubechies(N)


filter_bank = BuildBank(sp.constants.golden_ratio)
DaubAgaianWavelet = pywt.Wavelet(name = "DaubAgaianWavelet", filter_bank = filter_bank)


# noinspection PyTypeChecker
def enh_pipeline(image, sigma = 0.01, gaus_size = 3,
                 sharp_alpha = 1.1, scale = 255,
                 stretching_func = "dsigmoid", k = 1.0, b = 0.5, x_1 = 0.0, s = 900.0, alpha = 0.9):
    filtered = rescale(gaussian_filter(image = image, sigma = 0.01, size = gaus_size), 0, scale)
    sharpened = rescale(image, 0, scale) + sharp_alpha * (rescale(image, 0, scale) - filtered)

    stretched = rescale(nonlinear_stretching(sharpened, func = stretching_func,
                                             k = k, b = b,
                                             x_1 = x_1, s = s), 0, scale)
    histogram_equalized = rescale(exposure.equalize_hist(stretched), 0, scale)
    coeffs = pywt.dwt2(histogram_equalized, DaubAgaianWavelet)
    LL = coeffs[0]
    (LH, HL, HH) = coeffs[1]
    LH = LH * (np.abs(LH) ** (1 - alpha))
    HL = HL * (np.abs(HL) ** (1 - alpha))  # ALNR_Filter()
    HH = HH * (np.abs(HH) ** (1 - alpha))  # pywt.threshold(HH,value=HH.mean(),mode='hard',substitute=0)
    LL = LL * (np.abs(LL) ** (1 - alpha))
    enhanced = pywt.idwt2(coeffs = (LL, (LH, HL, HH)), wavelet = DaubAgaianWavelet)
    return enhanced


def alpha_pyramid_reduce(image, d, win_h, win_w, downscale = 2, max_level = 4):
    exception = BaseException("Maximal depth level too big!")
    if max_level > np.ceil(np.log2(image.shape[0])):
        raise exception
    result = []
    downscaled = image
    result.append(downscaled)
    for level in range(max_level):
        smoothed = alpha_trimmed(image = downscaled, d = d, win_h = win_h, win_w = win_w)
        downscaled = downscale_local_mean(smoothed, factors = (downscale, downscale))
        result.append(downscaled)
    return result


def get_patches(index, draw = True):
    indices_p = list(np.where(target == 1))[0]
    indices_h = list(np.where(target == 0))[0]
    indices_p = indices_p[:min(len(indices_h), len(indices_p))]
    indices_h = indices_h[:min(len(indices_h), len(indices_p))]

    image_healthy = images[indices_h[index]]
    image_pneumonia = images[indices_p[index]]

    good_patches = []
    bad_patches = []
    patient = patients[patients["id"] == ids_[indices_p[index]]]
    for i in range(len(patient)):
        x = int(patient.iloc[i].x)
        y = int(patient.iloc[i].y)
        w = int(patient.iloc[i].width)
        h = int(patient.iloc[i].height)
        good_patches.append(rescale(np.asarray(image_healthy[y:y + h, x:x + w, ].ravel().tolist()).reshape((h, w)),
                                    0, 1))
        bad_patches.append(rescale(np.asarray(image_pneumonia[y:y + h, x:x + w, ].ravel().tolist()).reshape((h, w)),
                                   0, 1))
    if draw:
        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 8))
        ax[0].imshow(image_healthy)
        ax[0].set_title("Healthy")
        ax[1].imshow(image_pneumonia)
        ax[1].set_title("Pneumonic")
        for i in range(len(patient)):
            x = int(patient.iloc[i].x)
            y = int(patient.iloc[i].y)
            w = int(patient.iloc[i].width)
            h = int(patient.iloc[i].height)
            ax[0].plot([x, x + w], [y, y], color = 'r')
            ax[0].plot([x + w, x + w], [y, y + h], color = 'r')
            ax[0].plot([x, x], [y, y + h], color = 'r')
            ax[0].plot([x, x + w], [y + h, y + h], color = 'r')
            ax[1].plot([x, x + w], [y, y], color = 'r')
            ax[1].plot([x + w, x + w], [y, y + h], color = 'r')
            ax[1].plot([x, x], [y, y + h], color = 'r')
            ax[1].plot([x, x + w], [y + h, y + h], color = 'r')
    return good_patches, bad_patches


# noinspection PyTypeChecker
def collect_features_(image,
                      distances = [1, 2, 3],
                      levels = list(range(3)),
                      n_points = 8, radius = 1, angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                      target = None):
    lengths = []

    indices_p = list(np.where(target == 1))[0]
    indices_h = list(np.where(target == 0))[0]
    indices_p = indices_p[:min(len(indices_h), len(indices_p))]
    indices_h = indices_h[:min(len(indices_h), len(indices_p))]

    if image.mean() > 120:
        image = image.max() - image
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    all_feat_g = 1
    image = rescale(image, 0, 255)
    # collect gaussian pyramids
    p_ = tuple(pyramid_gaussian(image = image, multichannel = False, max_layer = levels[-1]))
    for each_level in levels:
        glcm = greycomatrix(rescale(p_[each_level], 0, 255).astype('uint8'), distances, angles,
                            256, symmetric = True, normed = True)
        feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
        lbp = local_binary_pattern(p_[each_level], n_points, radius)
        lbp_hist, _ = np.histogram(lbp, density = False)
        coeffs2 = pywt.dwt2(p_[each_level], wavelet = DaubAgaianWavelet)
        LL, (LH, HL, HH) = coeffs2
        LH_hist, _ = np.histogram(LH, density = False)  # get_histogram(LH)
        HH_hist, _ = np.histogram(HH, density = False)  # get_histogram(HH)
        HL_hist, _ = np.histogram(HL, density = False)  # get_histogram(HL)
        all_feat_g = all_feat_g * np.concatenate([feats,
                                                  lbp_hist,
                                                  LH_hist,  # std_sc(LH_hist),
                                                  HH_hist,  # std_sc(HH_hist),
                                                  HL_hist])  # std_sc(HL_hist)])
    # return all_feat, lengths, np.hstack([std_sc(LH_hist), std_sc(HH_hist), std(HL_hist)]).shape[0]
    all_feat_g = np.cbrt(all_feat_g)
    all_feat_l = 1
    p_ = tuple(pyramid_laplacian(image = image, multichannel = False, max_layer = levels[-1] + 1))
    for each_level in levels:
        glcm = greycomatrix(rescale(p_[each_level], 0, 255).astype('uint8'), distances, angles,
                            256, symmetric = True, normed = True)
        feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])

        lbp = local_binary_pattern(p_[each_level], n_points, radius)
        lbp_hist, _ = np.histogram(lbp, density = True)

        coeffs2 = pywt.dwt2(p_[each_level], wavelet = DaubAgaianWavelet)
        LL, (LH, HL, HH) = coeffs2
        LH_hist, _ = np.histogram(LH, density = False)  # get_histogram(LH)
        HH_hist, _ = np.histogram(HH, density = False)  # get_histogram(HH)
        HL_hist, _ = np.histogram(HL, density = False)  # get_histogram(HL)

        all_feat_l = all_feat_l * np.concatenate([feats,
                                                  lbp_hist,  # std_sc(),
                                                  LH_hist,  # std_sc(),
                                                  HH_hist,  # std_sc(),
                                                  HL_hist])  # std_sc()])

    all_feat_l = np.cbrt(all_feat_l)
    all_feat_alpha = 1
    p_ = alpha_pyramid_reduce(image = image, d = 5, win_h = 5, win_w = 5, downscale = 2, max_level = levels[-1] + 1)
    for each_level in levels:
        glcm = greycomatrix(rescale(p_[each_level], 0, 255).astype('uint8'), distances, angles,
                            256, symmetric = True, normed = True)
        feats = np.hstack([std_sc(greycoprops(glcm, prop).ravel()) for prop in properties])

        lbp = local_binary_pattern(p_[each_level], n_points, radius)
        lbp_hist, _ = np.histogram(lbp, density = True)

        coeffs2 = pywt.dwt2(p_[each_level], wavelet = DaubAgaianWavelet)
        LL, (LH, HL, HH) = coeffs2
        LH_hist, _ = np.histogram(LH, density = True)  # get_histogram(LH)
        HH_hist, _ = np.histogram(HH, density = True)  # get_histogram(HH)
        HL_hist, _ = np.histogram(HL, density = True)  # get_histogram(HL)

        all_feat_alpha = all_feat_alpha * np.concatenate([feats,
                                                          lbp_hist,  # std_sc(lbp_hist),
                                                          LH_hist,  # std_sc(LH_hist),
                                                          HH_hist,  # std_sc(HH_hist),
                                                          HL_hist])  # std_sc(HL_hist)])

    all_feat_alpha = np.cbrt(all_feat_alpha)
    all_feat = np.concatenate([all_feat_g, all_feat_l, all_feat_alpha])
    return all_feat


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


if __name__ == '__main__':

    images = np.load("modules/data/images.npz")['arr_0']  # "lungs.npy");
    ids_ = np.load("modules/data/ids.npy")
    target = np.load("modules/data/target.npy")
    patients = pd.read_csv("modules/data/patients.csv")

    # Enhancement of all images
    for i in range(len(images)):
        images[i] = enh_pipeline(images[i], gaus_size = 5, sharp_alpha = 1.5,
                                 alpha = 0.98, scale = 1, k = 0.01, s = 9, b = 0.1, x_1 = 1)

    indices_p = list(np.where(target == 1))[0]
    indices_h = list(np.where(target == 0))[0]
    indices_p = indices_p[:min(len(indices_h), len(indices_p))]
    indices_h = indices_h[:min(len(indices_h), len(indices_p))]

    all_patch_features_h = np.empty((1, 336))
    all_patch_features_p = np.empty((1, 336))

    for index in range(len(indices_h)):
        h_patches, p_patches = get_patches(index, draw = False)
        for i in range(len(h_patches)):
            patch_h = h_patches[i]
            patch_p = p_patches[i]
            f_h = collect_features_(patch_h)
            f_p = collect_features_(patch_p)
            all_patch_features_h = np.vstack([all_patch_features_h, f_h])
            all_patch_features_p = np.vstack([all_patch_features_p, f_p])

    patch_targets = np.asarray([0] * 1168 + [1] * 1168)
    all_patch_features_ = np.vstack([all_patch_features_h[1:, ...], all_patch_features_p[1:, ...]])

    all_patch_features_, patch_targets = unison_shuffled_copies(all_patch_features_, patch_targets)

    del images

    names_ = np.load("feature_names.npy")

    X = all_patch_features_  # np.load("all_patch_features_.npy")  # unscaled features
    Y = patch_targets  # np.load("patch_targets_.npy")

    del all_patch_features_, patch_targets

    scaler = MinMaxScaler()  # StandardScaler()
    X_ = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X_, Y, test_size = 0.15)
    train_accs = []
    test_accs = []

    for feature in range(336):
        svm = SVC(kernel = 'rbf',
                  C = 2,
                  # degree = 5,
                  gamma = 'scale',
                  tol = 1e-6, probability = True)
        model = svm.fit(x_train[:, feature].reshape((-1, 1)), y_train)
        train_accs.append(model.score(x_train[:, feature].reshape((-1, 1)), y_train))
        test_accs.append(model.score(x_test[:, feature].reshape((-1, 1)), y_test))
        if feature % 50 == 0:
            print(feature)
    plt.figure(figsize = (10, 8))
    plt.plot(train_accs, label = "Training accuracy")
    plt.plot(test_accs, label = "Testing accuracy")
    plt.legend()
    plt.xlabel("Features")
    plt.ylabel("Accuracy")
    plt.savefig("feature_plots/individual_accs")

    train_dict = {}

    for each_feature in names_:
        train_dict[each_feature] = train_accs[names_.tolist().index(each_feature)]

    test_dict = {}
    for each_feature in names_:
        test_dict[each_feature] = test_accs[names_.tolist().index(each_feature)]
    import operator

    sorted_train_dict = dict(sorted(train_dict.items(), key = operator.itemgetter(1), reverse = True))
    sorted_test_dict = dict(sorted(test_dict.items(), key = operator.itemgetter(1), reverse = True))
    plt.figure(figsize = (60, 30))
    plt.xticks(ticks = list(range(336)), labels = list(sorted_train_dict.keys()), rotation = 45);
    plt.plot(sorted_train_dict.values());
    plt.savefig("feature_contributions")

    sorted_names = list(sorted_train_dict.keys())  # feature names sorted by training accuracy (descending)
    ind_feats = []
    train_accs_ = []
    test_accs_ = []
    for feature in range(336):
        ind_feats.append(names_.tolist().index(sorted_names[feature]))
        svm = SVC(kernel = 'rbf',
                  C = 2,
                  # degree = 5,
                  gamma = 'scale',
                  tol = 1e-6, probability = True)
        model = svm.fit(x_train[:, ind_feats].reshape((-1, len(ind_feats))), y_train)
        train_accs_.append(model.score(x_train[:, ind_feats].reshape((-1, len(ind_feats))), y_train))
        test_accs_.append(model.score(x_test[:, ind_feats].reshape((-1, len(ind_feats))), y_test))
        if feature % 50 == 0:
            print(feature)

    # Cumulative feature analysis
    sorted_names = list(sorted_train_dict.keys())  # feature names sorted by training accuracy (descending)
    ind_feats = []
    train_accs_ = []
    test_accs_ = []
    for feature in range(336):
        ind_feats.append(names_.tolist().index(sorted_names[feature]))
        svm = SVC(kernel = 'rbf',
                  C = 2,
                  # degree = 5,
                  gamma = 'scale',
                  tol = 1e-6, probability = True)
        model = svm.fit(x_train[:, ind_feats].reshape((-1, len(ind_feats))), y_train)
        train_accs_.append(model.score(x_train[:, ind_feats].reshape((-1, len(ind_feats))), y_train))
        test_accs_.append(model.score(x_test[:, ind_feats].reshape((-1, len(ind_feats))), y_test))
        if feature % 50 == 0:
            print(feature)
    plt.figure(figsize = (16, 8))
    plt.xlabel("Feature number")
    plt.ylabel("Accuracy")
    plt.xticks(ticks = list(range(1, 336, 6)), labels = names_[1:336:6], rotation = 45)
    plt.plot(train_accs_, label = "Training accuracy after adding features cumulatively");
    plt.plot(test_accs_, label = "Testing accuracy after adding features cumulatively");
    plt.legend()
    plt.savefig("cumulative_feature_combination_336_1")

    ind_feats = []
    train_accs_ = [0]
    test_accs_ = [0]
    feature = 0
    tr_acc_new = 0
    test_acc_new = 0

    while feature < 336:
        # for feature in range(336):
        ind_feats.append(names_.tolist().index(sorted_names[feature]))
        svm = SVC(kernel = 'rbf',
                  C = 2,
                  # degree = 5,
                  gamma = 'scale',
                  tol = 1e-6, probability = True)
        model = svm.fit(x_train[:, ind_feats].reshape((-1, len(ind_feats))), y_train)
        tr_acc_new = model.score(x_train[:, ind_feats].reshape((-1, len(ind_feats))), y_train)
        test_acc_new = model.score(x_test[:, ind_feats].reshape((-1, len(ind_feats))), y_test)
        if (tr_acc_new - train_accs_[-1]) > 0 and (
                test_acc_new - test_accs_[-1]) > 0:  # if training_acc & testing_acc improved
            train_accs_.append(tr_acc_new)
            test_accs_.append(test_acc_new)
            feature += 1
        else:
            ind_feats.remove(names_.tolist().index(sorted_names[feature]))
            feature += 1
        if feature % 50 == 0:
            print(feature)

    xticks_ = [names_.tolist().index(sorted_names[feature]) for feature in ind_feats]
    np.array(sorted_names)[xticks_].tolist()
    plt.figure(figsize = (16, 8))
    plt.xlabel("Feature number")
    plt.ylabel("Accuracy")
    plt.xticks(ticks = range(len(ind_feats)), labels = np.array(sorted_names)[xticks_].tolist(), rotation = 45)
    plt.plot(train_accs_[1:], label = "Training accuracy after adding features cumulatively")
    plt.plot(test_accs_[1:], label = "Testing accuracy after adding features cumulatively")
    plt.legend()
    plt.savefig("final_accuracies")

    # Perform KFold validation
    x_train_, x_test_, y_train, y_test = train_test_split(X_[:, xticks_], Y, test_size = 0.15)

    # train svm with 5-fold cross validation with radial basis kernel
    svm = SVC(kernel = 'rbf',
              C = 2,
              # degree = 5,
              gamma = 'scale',
              tol = 1e-6, probability = True)

    kfold = KFold(n_splits = 3)
    fold = 0
    f1 = []
    acc_scores = []
    roc_auc = []
    prec = []
    # x_train_ = x_train[:,np.array(xticks_)]
    # x_test_ = x_test[:,np.array(xticks_)]
    for train_index, test_index in kfold.split(x_train_):
        fold += 1
        model = svm.fit(x_train_[train_index], y_train[train_index])
        print(
            f"Fold: {fold};\nTraining accuracy: {100 * model.score(x_train_[train_index], y_train[train_index]):.3f}%")
        y_pred = model.predict(x_train_[test_index])
        print(f"F-1 score: {100 * f1_score(y_pred = y_pred, y_true = y_train[test_index]):.3f}%")
        print(f"Accuracy score: {100 * accuracy_score(y_pred = y_pred, y_true = y_train[test_index]):.3f}%")
        print(f"ROC-AUC score: {roc_auc_score(y_true = y_train[test_index], y_score = model.predict_proba(x_train_[test_index])[:, 1]):.3f}")
        print(f"Precision score: {100 * precision_score(y_true = y_train[test_index], y_pred = y_pred):.3f}\n\n")
        f1.append(f1_score(y_pred = y_pred, y_true = y_train[test_index]))
        acc_scores.append(accuracy_score(y_pred = y_pred, y_true = y_train[test_index]))
        roc_auc.append(roc_auc_score(y_true = y_train[test_index], y_score = y_pred))
        prec.append(precision_score(y_true = y_train[test_index], y_pred = y_pred))

    y_pred = model.predict(x_test_)
    print(f"Test set scores:\n")
    print(f"F-1 score: {100 * f1_score(y_pred = y_pred, y_true = y_test):.3f}%")
    print(f"Accuracy score: {100 * accuracy_score(y_pred = y_pred, y_true = y_test):.3f}")
    print(f"ROC-AUC score: {roc_auc_score(y_true = y_test, y_score = model.predict_proba(x_test_)[:, 1]):.3f}")
    print(f"Precision score: {100 * precision_score(y_true = y_test, y_pred = y_pred):.3f}\n\n")
    plt.figure(figsize = (10, 8))
    sns.heatmap(confusion_matrix(y_pred, y_test), annot = True, fmt = "d", annot_kws = {"size": 16})
    plt.figure(figsize = (10, 8))
    names = ["F1 Score", "Accuracy score", "Area-under-Curve score", "Precision"]
    for idx, score in enumerate([f1, acc_scores, roc_auc, prec]):
        # plt.figure(figsize = (10,8))
        # plt.title(f"{names[idx]} plot")
        plt.plot(range(1, kfold.n_splits + 1), score, label = f"{names[idx]} plot")
        plt.xlabel("Fold")
        plt.ylabel(names[idx])
        plt.legend()

    plt.figure(figsize = (10, 8))
    plt.plot(roc_curve(y_true = y_test, y_score = model.predict_proba(x_test_)[:, 1])[0],
             label = 'False-positive rates')
    plt.plot(roc_curve(y_true = y_test, y_score = model.predict_proba(x_test_)[:, 1])[1], label = 'True-positive rates')
    plt.legend()
    plt.savefig('roc_curve')

    plt.figure(figsize = (10, 8))
    plt.plot(precision_recall_curve(y_true = y_test, probas_pred = model.decision_function(x_test_))[0],
             label = "Precision")
    plt.plot(precision_recall_curve(y_true = y_test, probas_pred = model.decision_function(x_test_))[1],
             label = "Recall")
    plt.legend()
    plt.savefig('precision_recall_curve')
