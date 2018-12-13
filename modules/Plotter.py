import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#class Plotter:
"""
    def __init__(self, index,
                 images,
                 patients,
                 ids_, scale=False,
                 new_shape=None):
        self.index = index
        self.images = images
        self.patients = patients
        self.ids_ = ids_
        self.scale = scale
        self.new_shape = new_shape
"""


def plotImg(image,
            indx,
            imgs,
            patnts,
            ids,
            f_size=(6, 6),
            bbox=False,
            scaled=True,
            new_shape=None):
    plt.figure(figsize=f_size)
    plt.imshow(image)
    #if bbox:
    patient = patnts[patnts["id"] == ids[indx]].fillna(axis=1, value=0)

    for i in range(len(patient.index)):
        x = patient.iloc[i].x / 2#(imgs[indx].shape[0] / new_shape[0])
        y = patient.iloc[i].y / 2#(imgs[indx].shape[1] / new_shape[1])
        w = patient.iloc[i].width /2# (imgs[indx].shape[0] / new_shape[0])
        h = patient.iloc[i].height / 2#(imgs[indx].shape[1] / new_shape[1])
        plt.plot([x, x + w], [y, y], color='r')
        plt.plot([x + w, x + w], [y, y + h], color='r')
        plt.plot([x, x], [y, y + h], color='r')
        plt.plot([x, x + w], [y + h, y + h], color='r')
    