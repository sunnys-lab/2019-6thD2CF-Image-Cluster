"""
이미지 폴더의 파일을 분석하여 예측 레이블(labels_pred)을 생성하는 모듈
"""
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import os


from somber import Som
from evaluation import *
from config import *
from numpy import (array, unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, arange, outer, meshgrid, dot,
                   logical_and, mean, std, cov, argsort, linspace, transpose)

"""Returns the distance map of the weights.
Each cell is the normalised sum of the distances between
a neuron and its neighbours."""
def distance_map(self):
    um = zeros((self.shape[0], self.shape[1]))
    it = nditer(um, flags=['multi_index'])
    while not it.finished:
        for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):
            for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):
                if (ii >= 0 and ii < self.shape[0] and
                        jj >= 0 and jj < self.shape[1]):
                    w_1 = self[ii, jj, :]
                    w_2 = self[it.multi_index]
                    um[it.multi_index] += fast_norm(w_1 - w_2)
        it.iternext()
    um = um / um.max()
    return um


"""Returns norm-2 of a 1-D numpy array.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
"""
def fast_norm(x):

    return math.sqrt(dot(x, x.T))



"""
Self-Organizing Maps (SOMS) 알고리즘으로 특징 벡터를 클러스터링 하는 함수
예측 레이블은 DATA_DIR/LABELS_PRED.npy 에 저장
:return: None
"""
def make_labels_pred():

    # 01. Load datasets
    X = features = np.load(os.path.join(DATA_DIR, FEATURES + ".npy"))
    data_dim = X.shape[1]

    # 02. Estimate SOM matrix dimension
    estimate_max_cluster = int((len(features) / NUM_IMGS_PER_MODEL)*2.0)
    matrix_dim = round(math.sqrt(estimate_max_cluster))

    # 03. Data normalizing
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(-1, 1))
    X = sc.fit_transform(X)

    # 04. Training the SOM
    som = Som((matrix_dim, matrix_dim), data_dimensionality=data_dim, learning_rate=0.5, lr_lambda=2.5, infl_lambda=2.5)
    som.fit(X, num_epochs=10, updates_epoch=10, show_progressbar=True)

    # predict: get the index of each best matching unit.
    labels_pred = som.predict(X)

    # save predicted labels
    np.save(os.path.join(DATA_DIR, LABELS_PRED + ".npy"), labels_pred)
    np.savetxt(os.path.join(DATA_DIR, LABELS_PRED + ".tsv"), labels_pred, "%d", delimiter="\t")
    np.savetxt(os.path.join(DATA_DIR, LABELS_PRED + ".txt"), labels_pred, "%d", delimiter="\t")

    # quantization error: how well do the best matching units fit?
    quantization_error = som.quantization_error(X)

    # inversion: associate each node with the exemplar that fits best.
    inverted = som.invert_projection(X, labels_pred)

    # Mapping: get weights, mapped to the grid points of the SOM
    mapped = som.map_weights()

    # 06.Visualization
    from pylab import bone, pcolor, colorbar, plot, show
    bone()
    distance = distance_map(mapped).T
    pcolor(distance)
    colorbar()
    show()


if __name__ == '__main__':
    make_labels_pred()