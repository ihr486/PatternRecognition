#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import mnread

def g(mean, icov, dcov, x):
    return -0.5 * (x - mean) * icov * (x - mean).T - 0.5 * dcov

class ModelBase:
    def __init__(self, label, data):
        self.mean = {}
        for x in set(label):
            self.mean[x] = np.mean(data[np.where(label==x),:],axis=1)

    def classify(self, data, icov, dcov):
        label = np.empty(data.shape[0],dtype=int)
        for i in range(data.shape[0]):
            gl = np.array([0]*10)
            for x in self.mean:
                gl[x] = g(self.mean[x], icov[x], dcov[x], np.matrix(data[i,:]))
            label[i] = np.argmax(gl)
        return label

class Model1(ModelBase):
    def __init__(self, label, data):
        ModelBase.__init__(self, label, data)
        self.icov = {}
        self.dcov = {}
        for x in set(label):
            self.icov[x] = np.identity(data.shape[1])
            self.dcov[x] = 0.0

    def classify(self, data):
        return ModelBase.classify(self, data, self.icov, self.dcov)

class Model2(ModelBase):
    def __init__(self, label, data):
        ModelBase.__init__(self, label, data)
        self.icov = {}
        self.dcov = {}
        data_ = np.empty_like(data)
        for x in set(label):
            data_[np.where(label==x)[-1],:] = data[np.where(label==x)[-1],:] - self.mean[x]
        cov_ = np.matrix(data_).T * np.matrix(data_) / float(data.shape[0]) + np.identity(data.shape[1])
        icov_ = np.linalg.inv(cov_)
        sgn_,dcov_ = np.linalg.slogdet(cov_)
        for x in set(label):
            self.icov[x] = icov_
            self.dcov[x] = dcov_

    def classify(self, data):
        return ModelBase.classify(self, data, self.icov, self.dcov)

class Model3(ModelBase):
    def __init__(self, label, data):
        ModelBase.__init__(self, label, data)
        self.icov = {}
        self.dcov = {}
        for x in set(label):
            data_ = data[np.where(label == x)[-1],:]
            cov_ = np.matrix(data_ - self.mean[x]).T * np.matrix(data_ - self.mean[x]) / float(data_.shape[0]) + np.identity(data.shape[1])
            self.icov[x] = np.linalg.inv(cov_)
            sgn_,dcov_ = np.linalg.slogdet(cov_)
            self.dcov[x] = dcov_

    def classify(self, data):
        return ModelBase.classify(self, data, self.icov, self.dcov)

if __name__ == "__main__":
    trlabel = mnread.readlabel(mnread.trlabelfz)
    trdata = mnread.readim(mnread.trdatafz)
    tstlabel = mnread.readlabel(mnread.tstlabelfz)
    tstdata = mnread.readim(mnread.tstdatafz)

    trdataF = np.reshape(trdata, [trdata.shape[0],-1])
    tstdataF = np.reshape(tstdata, [tstdata.shape[0],-1])

    model = Model1(trlabel, trdataF)
    #model = Model2(trlabel, trdataF)
    #model = Model3(trlabel, trdataF)
    estlabel = model.classify(tstdataF)
    print('accuracy: %g' % (float(sum(estlabel==tstlabel)) / len(tstlabel)))

    plt.figure()
    plt.suptitle('goods')
    goods = np.random.permutation(np.where(estlabel==tstlabel)[-1])[range(50)]
    for i,good in enumerate(goods):
        plt.subplot(5,10,i+1)
        plt.axis('off')
        plt.imshow(tstdata[good,:,:],cmap='gray')
        plt.title(estlabel[good])
    plt.figure()
    plt.suptitle('bads')
    bads = np.random.permutation(np.where(~(estlabel==tstlabel))[-1])[range(50)]
    for i,bad in enumerate(bads):
        plt.subplot(5,10,i+1)
        plt.axis('off')
        plt.imshow(tstdata[bad,:,:],cmap='gray')
        plt.title('%s(%s)' % (estlabel[bad], tstlabel[bad]))
    plt.show()
