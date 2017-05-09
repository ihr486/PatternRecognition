#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

N = 1000

def scale(v, s):
    m = np.diag(s)
    return v * m

def rotate(v, r):
    m = np.matrix([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return v * m

def g(mean, cov, P, xx, yy):
    icov = np.linalg.inv(cov)
    xxa = np.array(xx) - mean[0, 0]
    yya = np.array(yy) - mean[0, 1]
    return -0.5 * (icov[0,0] * xxa * xxa + (icov[0,1] + icov[1,0]) * xxa * yya + icov[1,1] * yya * yya) - 0.5 * np.log(np.linalg.det(cov)) + np.log(P)

def norm_dist(mean, cov, xx, yy):
    icov = np.linalg.inv(cov)
    xxa = np.array(xx) - mean[0, 0]
    yya = np.array(yy) - mean[0, 1]
    return 0.5 / (np.pi * np.sqrt(np.linalg.det(cov))) * np.exp(-0.5 * (icov[0,0] * xxa * xxa + (icov[0,1] + icov[1,0]) * xxa * yya + icov[1,1] * yya * yya))

if __name__ == "__main__":
    x0 = rotate(scale(np.matrix(np.random.randn(N, 2)), [1.3, 1]), np.pi * 0.35) + [0, 3]
    x1 = rotate(scale(np.matrix(np.random.randn(N, 2)), [1.3, 1]), np.pi * 0.35) + [4, 0]
    #x1 = rotate(scale(np.matrix(np.random.randn(N, 2)), [1.7, 1]), np.pi * 0.7) + [4, 0]

    P0 = 0.5
    P1 = 1 - P0

    m0 = np.mean(x0, axis=0)
    m1 = np.mean(x1, axis=0)

    cov0 = (x0 - m0).T * (x0 - m0) / N
    cov1 = (x1 - m1).T * (x1 - m1) / N

    l0 = (g(m0, cov0, P0, x0.T.A[0], x0.T.A[1]) >= g(m1, cov1, P1, x0.T.A[0], x0.T.A[1]))
    l1 = (g(m0, cov0, P0, x1.T.A[0], x1.T.A[1]) >= g(m1, cov1, P1, x1.T.A[0], x1.T.A[1]))

    plt.figure()
    plt.plot(x0[np.where(l0),0].ravel(), x0[np.where(l0),1].ravel(), 'bo', zorder=1)
    plt.plot(x0[np.where(~l0),0].ravel(), x0[np.where(~l0),1].ravel(), 'ro', zorder=1)
    plt.plot(x1[np.where(l1),0].ravel(), x1[np.where(l1),1].ravel(), 'r^', zorder=1)
    plt.plot(x1[np.where(~l1),0].ravel(), x1[np.where(~l1),1].ravel(), 'b^', zorder=1)
    plt.axis('scaled')
    mgx, mgy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1]), np.linspace(plt.ylim()[0], plt.ylim()[1]))
    plt.contour(mgx, mgy, norm_dist(m0, cov0, mgx, mgy), cmap='hsv', zorder=2)
    plt.contour(mgx, mgy, norm_dist(m1, cov1, mgx, mgy), cmap='hsv', zorder=2)
    plt.clabel(plt.contour(mgx, mgy, g(m0, cov0, P0, mgx, mgy) - g(m1, cov1, P1, mgx, mgy), cmap='hsv', zorder=3), zorder=4)
    plt.show()
