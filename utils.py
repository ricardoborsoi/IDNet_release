#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contains utilities to plot images and do related stuff

Created on Fri Aug  6 21:05:58 2021

@author: ricardo
"""

from matplotlib import pyplot as plt
import torch

def reshape_fortran(x, shape):
    ''' perform a reshape in Matlab/Fortran style '''
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def plotImage(dataY, nr, nc, L=None):
    '''plots an image, a few bands (L by N)'''
    # if isinstance(dataY, list):
        # Y = torch.zeros((nr,nc,L))
    if L == None:
        L = dataY.shape[0]
    plt.figure()
    Yim = torch.reshape(dataY.T, (nc,nr,L))
    plt.imshow(Yim[:,:,[10, 20, 30]])
    plt.show()


def plotEMs(M, thetitle='title'):
    P = M.shape[1] # number of endmembers
    L = M.shape[0] # number of bands
    fig = plt.figure()
    for i in range(P):
        plt.plot(torch.linspace(1,L,L), M[:,i])
    plt.title(thetitle, fontsize=12)
    plt.show()


def plotAbunds(A, nr, nc, thetitle='title', savepath=None):
    ''' plots abundance maps, should be P by N '''
    P = A.shape[0] # number of endmembers
    N = A.shape[1] # number of pixels
    A_cube = torch.reshape(A.T, (nc,nr,P))
    fig, axs = plt.subplots(1, P)
    for i in range(P):
        axs[i].imshow(A_cube[:,:,i].T, cmap='jet', vmin=0, vmax=1) #cmap='gray'
        axs[i].axis('off')
    # plt.axis('off')
    # axs[P-1].colorbar()
    # fig = plt.figure()
    # # plt.imshow(temp)
    # plt.imshow(A_cube[:,:,0], cmap='gray', vmin=0, vmax=1)
    # plt.colorbar()
    plt.title(thetitle, fontsize=12)
    if savepath is not None: # save a figure is specified
        plt.savefig(savepath, dpi=300, format='pdf')
    plt.show()


def show_ground_truth(A_true, Mgt_avg, nr, nc):
    plotEMs(Mgt_avg, thetitle='ground truth')
    plotAbunds(A_true, nr=nr, nc=nc, thetitle='ground truth')


def compute_metrics(t_true, t_est):
    RMSE = torch.sqrt(torch.sum((t_true-t_est)**2)/t_true.shape.numel())
    NRSME = torch.sqrt(torch.sum((t_true-t_est)**2)/torch.sum(t_true**2))
    return RMSE, NRSME


