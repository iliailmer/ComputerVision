import numpy as np


def kullback_leibler_div(proba_P, proba_Q):
    e = 0
    for i in range(len(proba_P)):
        if proba_P[i] > 0 and proba_Q[i] > 0:
            e += proba_P[i] * np.log2(proba_P[i] / proba_Q[i])
    return e


def mse(feat_h, feat_p):
    return np.abs(np.mean(np.asarray(feat_h) ** 2 - np.asarray(feat_p) ** 2))


def cosine_sim(feat_h, feat_p):
    return np.dot(np.asarray(feat_h), np.asarray(feat_p)) / (np.linalg.norm(np.asarray(feat_h)) * np.linalg.norm(np.asarray(feat_p)))


def mean(image):
    return np.mean(image)


def std(image):
    return np.std(image)


def median(image):
    return np.median(image)