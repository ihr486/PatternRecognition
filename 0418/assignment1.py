#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

N = 1000

def scale(v, x, y):
    m = np.matrix([[x, 0], [0, y]])
    return v * m

def translate(v, x, y):
    return v + np.array([x, y])

def rotate(v, r):
    m = np.matrix([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return v * m

if __name__ == "__main__":
    x0 = translate(rotate(scale(np.matrix(np.random.randn(N, 2)), 2, 0.7), np.pi*0.2), -1, 2)
    x1 = translate(rotate(scale(np.matrix(np.random.randn(N, 2)), 0.6, 1.4), np.pi*0.9), 2, -1)

    m0 = np.mean(x0, axis=0)
    m1 = np.mean(x1, axis=0)

    cov0 = (x0 - m0).T * (x0 - m0) / N
    cov1 = (x1 - m1).T * (x1 - m1) / N

    ed0, ev0 = np.linalg.eig(cov0)

    x0w = x0 * ev0 * np.diag(ed0 ** -0.5)
    x1w = x1 * ev0 * np.diag(ed0 ** -0.5)

    cov1w = np.diag(ed0 ** -0.5) * ev0.T * cov1 * ev0 * np.diag(ed0 ** -0.5)

    ed1w, ev1w = np.linalg.eig(cov1w)

    x0s = x0w * ev1w
    x1s = x1w * ev1w

    plt.figure()
    plt.scatter(x0.T.A[0], x0.T.A[1])
    plt.scatter(x1.T.A[0], x1.T.A[1])
    plt.axis('scaled')
    plt.figure()
    plt.scatter(x0w.T.A[0], x0w.T.A[1])
    plt.scatter(x1w.T.A[0], x1w.T.A[1])
    plt.axis('scaled')
    plt.figure()
    plt.scatter(x0s.T.A[0], x0s.T.A[1])
    plt.scatter(x1s.T.A[0], x1s.T.A[1])
    plt.axis('scaled')
    plt.show()
