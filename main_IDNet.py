#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:31:15 2021

@author: ricardo
"""


from __future__ import print_function
import math
import time
import argparse
from matplotlib import pyplot as plt
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.distributions as td
import DataGeneration as DG
import utils
from scipy.io import loadmat, savemat
import random
import numpy as np

# for reproducibility
# device = torch.device("cuda")
device = torch.device("cpu")
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)



class IDNet(nn.Module):
    ''' Class containing the whole architecture of ID-Net'''
    def __init__(self, P, L, H = 2):
        super(IDNet, self).__init__()
        
        # problem parameters ----------------------------------------
        self.P = P # number of endmembers
        self.H = H # latent EM space dimension
        self.L = L # number of bands
        
        # gains on the networks nonlinear parts ---------------------
        self.gain_nlin_a = 0.1
        self.gain_nlin_y = 0.1
        
        # (importance) sampling sizes -------------------------------
        self.K1 = 1
        self.K2 = 5
        
        
        
        # variables -------------------------------------------------
        # Variance of the pixel noise 
        self.fcy_std = nn.Linear(1, 1)
        
        # Parameters from the EM distribution, see also torch.nn.ParameterList()
        self.fcMz_mu  = torch.nn.ModuleList();
        self.fcMz_std = torch.nn.ModuleList();
        self.fcMz_std2 = torch.nn.ModuleList();
        for i in range(0, self.P):
            self.fcMz_mu.append(nn.Sequential(
                  nn.Linear(self.H, max(math.ceil(self.L/10),self.H+1)),
                  nn.ReLU(),
                  nn.Linear(max(math.ceil(self.L/10),self.H+1), max(math.ceil(self.L/4),self.H+2)+3),
                  nn.ReLU(),
                  nn.Linear(max(math.ceil(self.L/4),self.H+2)+3, math.ceil(1.2*self.L)+5),
                  nn.ReLU(),
                  nn.Linear(math.ceil(1.2*self.L)+5, self.L),
                  nn.Sigmoid() ))
            self.fcMz_std.append(nn.Sequential(
                  nn.Linear(self.H, 1),
                  nn.ReLU() ))
            self.fcMz_std2.append(nn.Linear(1, 1))
        
        
        # Parameters of the recognition model for Z|y
        self.fcZy_mu  = torch.nn.ModuleList();
        self.fcZy_std = torch.nn.ModuleList();
        for i in range(0, self.P):
            self.fcZy_mu.append(nn.Sequential(
                  nn.Linear(self.L, 5*self.H),
                  nn.ReLU(),
                  nn.Linear(5*self.H,2*self.H),
                  nn.ReLU(),
                  nn.Linear(2*self.H,self.H) ))
            
        for i in range(0, self.P): # share parameters for the covariance
            self.fcZy_std.append(nn.Sequential(
                  self.fcZy_mu[i][0],
                  nn.ReLU(),
                  self.fcZy_mu[i][2],
                  nn.ReLU(),
                  nn.Linear(2*self.H,2*self.H),
                  nn.ReLU(),
                  nn.Linear(2*self.H,2*self.H),
                  nn.ReLU(),
                  nn.Linear(2*self.H,self.H) ))
        
        
        
        
        
        # Precision of the abundance posterior distribution a|M,y
        self.fca_std = nn.Sequential(
                  nn.Linear(1, 1),
                  nn.ReLU())
        self.fca_std2 = nn.Linear(1, 1)
        
        
        # nonlinear part of the neural net that estimates the alphas (Dirichlet params)
        self.fca_My_alphas = nn.Sequential(
                  nn.Linear(self.L, 2*self.L),
                  nn.ReLU(),
                  nn.Linear(2*self.L, round(0.5*self.L)),
                  nn.ReLU(),
                  nn.Linear(round(0.5*self.L), round(0.25*self.L)),
                  nn.ReLU(),
                  nn.Linear(round(0.25*self.L), 4*self.P),
                  nn.ReLU(),
                  nn.Linear(4*self.P, self.P))
        

        
        self.fcy_Ma_nlin = nn.Sequential(
                  nn.Linear(self.P*(self.L+1), self.P*self.L),
                  nn.ReLU(),
                  nn.Linear(self.P*self.L, self.L),
                  nn.ReLU(),
                  nn.Linear(self.L, self.L),
                  nn.ReLU(),
                  nn.Linear(self.L, self.L))
        
        
        
        # related to abundance sparsity
        self.fca_llambda = nn.Linear(1, 1)
        self.K_iters_LISTA = 10; # number of LISTA iterations
        self.fca_llambda3 = nn.Linear(1, self.K_iters_LISTA)
        
        self.fca_Rtilde = nn.Sequential(
                  nn.Linear(self.P, self.P),
                  nn.ReLU())
        self.fca_prior = nn.Linear(1, self.P)
            
            
        
        
    def UnrolledNetAlphas(self, y, M, K_iter=10):
        # LISTA-type network ---------------------------------
        llambda  = 0.01*torch.exp(self.fca_llambda(torch.tensor([float(1)])))
        llambda3 = 0.01*torch.exp(self.fca_llambda3(torch.tensor([float(1)])))
        alpha = torch.bmm(torch.linalg.pinv(M), y.unsqueeze(2)).squeeze(2)
        for i in range(self.K_iters_LISTA):
            alpha = alpha - llambda3[i]*torch.bmm(M.permute(0,2,1), \
                    torch.bmm(M, alpha.unsqueeze(2))-y.unsqueeze(2)).squeeze(2)
            alpha = torch.nn.functional.relu(alpha - llambda3[i]*llambda, inplace=False) # soft shrinkage and project to nonnegative orthant
        alpha_hat = alpha
        return alpha_hat
    
    
    
    
    
    
    
    def compute_log_p_Gaussian(self, sigmas, mus, z):
        ''' computes log(p(z)), for p=N(mus,diag(sigmas)), with batched data
        sigmas : batch * K (variances, diagonal covariance matrix)
        mus : batch * K (means)
        z : batch * K (smaples) '''
        K = z.shape[1]
        exppart = - 0.5 * (((mus-z)/sigmas)*(mus-z)).sum(1)
        constpart = -0.5*K*torch.log(2*torch.tensor(math.pi)) \
                    -0.5*torch.log(sigmas).sum(1)
        return constpart + exppart
    
    
    
    def samp_q_Z_y(self, y, batch_size, K1):
        ''' returns a reparametrized sample from q(Z|y) in batch form
        y : batch_size * L (pixels)'''
        # compute means and covariances ------------
        Z_y_mean  = torch.zeros((batch_size*1,self.H,self.P))
        Z_y_sigma = torch.zeros((batch_size*1,self.H,self.P))
        for i in range(self.P):
            Z_y_mean[:,:,i] = self.fcZy_mu[i](y)
            Z_y_sigma[:,:,i] = torch.exp(self.fcZy_std[i](y)) # variances
        
        # replicate the means and variances for all MC samples (better than doing it on y)
        Z_y_mean = torch.kron(torch.ones((K1,1,1)), Z_y_mean)
        Z_y_sigma = torch.kron(torch.ones((K1,1,1)), Z_y_sigma)
        
        # sample epsilon from N(0,1)
        epsilon1 = torch.normal(torch.zeros((batch_size*K1,self.H,self.P)), 1)
        Z_y_samp = torch.sqrt(Z_y_sigma) * epsilon1 + Z_y_mean # reparametrization
        return Z_y_mean, Z_y_sigma, Z_y_samp
    
    
    def samp_q_M_Z(self, Z, batch_size, K1):
        ''' returns a reparametrized sample from q(M|Z) in batch form
        Z : (batch_sizeK1) * H * P (latent EM representations)'''
        M_Z_mean  = torch.zeros((batch_size*K1,self.L,self.P))
        M_Z_sigma = torch.zeros((batch_size*K1,self.L,self.P))
        for i in range(self.P):
            M_Z_mean[:,:,i] = self.fcMz_mu[i](Z[:,:,i]) # Z = Z_y_samp
            M_Z_sigma[:,:,i] = torch.ones((batch_size*K1,self.L)) * \
                        0.01*torch.exp(self.fcMz_std2[i](torch.tensor([float(1)])))
        
        # sample epsilon from N(0,1)
        epsilon2 = torch.normal(torch.zeros((batch_size*K1,self.L,self.P)), 1)
        M_Z_samp = torch.sqrt(M_Z_sigma) * epsilon2 + M_Z_mean # reparametrization
        return M_Z_mean, M_Z_sigma, M_Z_samp
    
    
    def samp_q_a_My(self, M, y, batch_size, K1, compute_nlin_deg=False):
        ''' returns a reparametrized sample from q(a|M,Z) in batch form
        M : (batch_sizeK1) * L * P (sampled EMs)
        y : batch_size * L (pixels)'''
        y = torch.kron(torch.ones((K1,1)), y)
        # compute alphas
        K = 10 + 100*torch.exp(self.fca_std2(torch.tensor([float(1)])))
        alphas_lin  = self.UnrolledNetAlphas(y, M) # lista, sparse linear part
        alphas_nlin = self.gain_nlin_a*self.fca_My_alphas(y) # nonlinear part
        alphas      = alphas_lin + alphas_nlin # add them together
        # project to nonnegative orthant:
        alphas_hat = 1e-16 + K*torch.nn.functional.relu(alphas, inplace=False) 
        
        # now sample the abundances
        q_a_My = []
        a_Zy_samp = torch.zeros((batch_size*K1,self.P))
        for i in range(batch_size*K1):
            q_a_My.append(td.Dirichlet(alphas_hat[i,:]))
            a_Zy_samp[i,:] = q_a_My[i].rsample()
        
        # if true, compute the rati of the energy of the nonlinear term in the abundance 
        # estimation compared to the energy of the full term
        if compute_nlin_deg is True:
            a_nlin_deg = torch.sqrt(torch.sum(alphas_nlin.detach()**2, dim=1)) \
                       /( torch.sqrt(torch.sum(alphas_lin.detach()**2, dim=1)) \
                        + torch.sqrt(torch.sum(alphas_nlin.detach()**2, dim=1)) )
            return q_a_My, alphas_hat, a_Zy_samp, a_nlin_deg
        
        return q_a_My, alphas_hat, a_Zy_samp
    
    
    
    def unmix(self, Y):
        ''' perform unmixing on the whole image Y (bands*pixels)'''
        N = Y.shape[1]
        M_avg, _, _ = self.samp_q_M_Z(Z=torch.zeros((1,self.H,self.P)), batch_size=1, K1=1)
        Z_y_mean, Z_y_sigma, Z_y_samp = self.samp_q_Z_y(Y.T, batch_size=N, K1=1)
        Mn_est, M_Z_sigma, M_Z_samp = self.samp_q_M_Z(Z=Z_y_samp, batch_size=N, K1=1)
        q_a_My, alphas, a_Zy_samp, a_nlin_deg = self.samp_q_a_My(M=Mn_est, y=Y.T, batch_size=N, K1=1, compute_nlin_deg=True)
        A_est = (alphas/torch.kron(torch.ones(1,self.P), alphas.sum(dim=1).unsqueeze(1))).T
        # compute the reconstructed image
        Y_rec = nn.functional.relu(torch.bmm(Mn_est, A_est.T.unsqueeze(2)).squeeze() \
            + self.gain_nlin_y*self.fcy_Ma_nlin(torch.cat((Mn_est.reshape((N,self.L*self.P)),A_est.T), dim=1))).T # nonlinear part
        return A_est, Mn_est.permute(1,2,0), M_avg.squeeze(), Y_rec, a_nlin_deg

    
    def forward(self, x_data):
        # x_data should have the supervised and semi-supervised part of the data (will be twiced)
        # it is a list, [unsup sup]
        # unsup is the bath_size * L pixels
        # sup is another list, [y M a], where y,M,a are bath_size * otherdims tensors
        
        # Get data -------------------------------
        batch_size = x_data[0].shape[0]
        
        y_unsup = x_data[0].to(device) # batch * L
        y_sup = x_data[1][0].to(device) # batch * L
        M_sup = x_data[1][1].to(device)
        a_sup = x_data[1][2].to(device)
        
                
        # construct latent PDFs and some parameters ------------------
        a_prior_par = torch.ones(self.P,)
        p_a = td.Dirichlet(a_prior_par)
        sigma_noise_y = 0.01*torch.exp(self.fcy_std(torch.tensor([float(1)]))) # variance of the pixels noise (per band)
        
        
        
        # first unsupervised part --------------------------------------------
        K1 = self.K1
        
        # initialize
        log_probs_unsup = dict()
        log_py_Ma = torch.zeros((batch_size*K1))
        log_pM_Z  = torch.zeros((batch_size*K1))
        log_pZ    = torch.zeros((batch_size*K1))
        log_pa    = torch.zeros((batch_size*K1))
        log_qZ_y  = torch.zeros((batch_size*K1))
        log_qM_Z  = torch.zeros((batch_size*K1))
        log_qa_My = torch.zeros((batch_size*K1))
        alphas_unsup = torch.zeros((self.P,batch_size*K1))
        
        # reparametrizeable sampling ---------------------------
        # first sample the Z, then from M, then from a
        Z_y_mean, Z_y_sigma, Z_y_samp = self.samp_q_Z_y(y_unsup, batch_size, K1)
        M_Z_mean, M_Z_sigma, M_Z_samp = self.samp_q_M_Z(Z_y_samp, batch_size, K1)
        q_a_My, alphas, a_Zy_samp = self.samp_q_a_My(M_Z_samp, y_unsup, batch_size, K1)

                
        # compute the parameters of p(y|M,a)
        y_Ma_sigma = sigma_noise_y * torch.ones((batch_size*K1,self.L))
        y_Ma_mean = torch.bmm(M_Z_samp, a_Zy_samp.unsqueeze(2)).squeeze() # linear part
        y_Ma_mean = y_Ma_mean + self.gain_nlin_y*self.fcy_Ma_nlin(torch.cat((M_Z_samp.reshape((batch_size*K1,self.L*self.P)),a_Zy_samp), dim=1)) # nonlinear part
        y_Ma_mean = nn.functional.relu(y_Ma_mean)
        
        
        # now compute the log-probabilities
        y_unsup_repl = torch.kron(torch.ones((K1,1)),y_unsup) # replicate across MC samples
        log_py_Ma = self.compute_log_p_Gaussian(y_Ma_sigma, y_Ma_mean, y_unsup_repl)
        for k in range(self.P):
            log_pM_Z = log_pM_Z + self.compute_log_p_Gaussian(M_Z_sigma[:,:,k], M_Z_mean[:,:,k], M_Z_samp[:,:,k]) # p_M_Z[k].log_prob(M_unsup[:,k])
            log_pZ   = log_pZ   + self.compute_log_p_Gaussian(torch.ones((batch_size*K1,self.H)), torch.zeros((batch_size*K1,self.H)), Z_y_samp[:,:,k]) # p_z.log_prob(Z_unsup[:,k])
            log_qZ_y = log_qZ_y + self.compute_log_p_Gaussian(Z_y_sigma[:,:,k], Z_y_mean[:,:,k], Z_y_samp[:,:,k]) # q_Z_y[k].log_prob(Z_unsup[:,k])
        log_qM_Z = log_pM_Z # we assume q=p for M|Z
        
        for i in range(batch_size*K1):
            log_qa_My[i] = q_a_My[i].log_prob(a_Zy_samp[i,:]+1e-16) 
            log_pa[i]    = p_a.log_prob(a_Zy_samp[i,:]+1e-16)
        
        # store alphas
        alphas_unsup = alphas.T
         
        
        # reorder (useless)
        log_py_Ma = utils.reshape_fortran(log_py_Ma, (batch_size,K1))
        log_pM_Z  = utils.reshape_fortran(log_pM_Z,  (batch_size,K1))
        log_pZ    = utils.reshape_fortran(log_pZ,    (batch_size,K1))
        log_pa    = utils.reshape_fortran(log_pa,    (batch_size,K1))
        log_qZ_y  = utils.reshape_fortran(log_qZ_y,  (batch_size,K1))
        log_qM_Z  = utils.reshape_fortran(log_qM_Z,  (batch_size,K1))
        log_qa_My = utils.reshape_fortran(log_qa_My, (batch_size,K1))
        alphas_unsup = utils.reshape_fortran(alphas_unsup, (self.P,batch_size,K1))
        
                
        # now store the log probs of the unspervised part
        log_probs_unsup = {'log_py_Ma':log_py_Ma, 'log_pM_Z':log_pM_Z,
                           'log_pZ':log_pZ, 'log_qZ_y':log_qZ_y,
                           'log_qM_Z':log_qM_Z, 'log_qa_My':log_qa_My, 'log_pa':log_pa}
        
        
        
        # now for the semi-supervised part ------------------------------------
        K2 = self.K2
        
        # initialize
        log_probs_sup = dict()
        log_py_Ma  = torch.zeros((batch_size*1)) # constant for MC samples K2
        log_pM_Z   = torch.zeros((batch_size*K2))
        log_pZ     = torch.zeros((batch_size*K2))
        log_pa     = torch.zeros((batch_size*1)) # constant for MC samples K2
        log_qZ_y   = torch.zeros((batch_size*K2))
        log_qM_Z   = torch.zeros((batch_size*K2))
        log_qa_My  = torch.zeros((batch_size*1)) # constant for MC samples K2
        log_omegas = torch.zeros((batch_size,K2)) # the log-importante weights
        alphas_sup = torch.zeros((self.P,batch_size*K2))
        
        
        # reparametrizeable sampling ---------------------------
        # first sample the Z, then from M, then from a
        Z_y_mean, Z_y_sigma, Z_y_samp = self.samp_q_Z_y(y_sup, batch_size, K2)
        M_Z_mean, M_Z_sigma, M_Z_samp = self.samp_q_M_Z(Z_y_samp, batch_size, K2)
        q_a_My, alphas, a_Zy_samp = self.samp_q_a_My(M_sup, y_sup, batch_size, 1) # compute once and replicate later

        
        # compute the parameters of p(y|M,a)
        y_Ma_sigma = sigma_noise_y * torch.ones((batch_size,self.L))
        y_Ma_mean = torch.bmm(M_sup, a_sup.unsqueeze(2)).squeeze() # linear part
        y_Ma_mean = y_Ma_mean + self.gain_nlin_y*self.fcy_Ma_nlin(torch.cat((M_sup.reshape((batch_size,self.L*self.P)),a_sup), dim=1)) # nonlinear part
        y_Ma_mean = nn.functional.relu(y_Ma_mean)
        
        
        # replicate M_sup to match MC samples
        M_sup_repl = torch.kron(torch.ones((K2,1,1)), M_sup)
        
        # now compute the log-probabilities
        log_py_Ma = self.compute_log_p_Gaussian(y_Ma_sigma, y_Ma_mean, y_sup)
        for k in range(self.P):
            log_pM_Z = log_pM_Z + self.compute_log_p_Gaussian(M_Z_sigma[:,:,k], M_Z_mean[:,:,k], M_sup_repl[:,:,k]) # p_M_Z[k].log_prob(M_sup[:,k])
            log_pZ   = log_pZ   + self.compute_log_p_Gaussian(torch.ones((batch_size*K2,self.H)), torch.zeros((batch_size*K2,self.H)), Z_y_samp[:,:,k]) # p_z.log_prob(Z_sup[:,k])
            log_qZ_y = log_qZ_y + self.compute_log_p_Gaussian(Z_y_sigma[:,:,k], Z_y_mean[:,:,k], Z_y_samp[:,:,k]) # q_Z_y[k].log_prob(Z_sup[:,k])
        log_qM_Z = log_pM_Z # we assume q=p for M|Z
        
        for i in range(batch_size*1):
            log_qa_My[i] = q_a_My[i].log_prob(a_Zy_samp[i,:]+1e-16) 
            log_pa[i]    = p_a.log_prob(a_Zy_samp[i,:]+1e-16)
        
        
        # replicate log_p values that were constant across MC samples K2
        log_py_Ma = torch.kron(torch.ones((K2,1)), log_py_Ma)
        log_qa_My = torch.kron(torch.ones((K2,1)), log_qa_My)
        log_pa    = torch.kron(torch.ones((K2,1)), log_pa)
        
        # store alphas
        alphas_sup = torch.kron(torch.ones((K2,1)), alphas).T
        
        
        # now compute the log-importance weights
        log_omegas = utils.reshape_fortran(log_qM_Z, (batch_size,K2))
        
        # for k in range(0,self.P):
        #     log_omegas[i,j] = log_omegas[i,j] + q_M_Z[k].log_prob(M_sup[i,:,k]) # compute log[ q(M(i)|Z(j)) ]
        # #omegas[i,j] = torch.exp(omegas[i,j]) # compute q(M(i)|Z(j)) --- don't do this to void numerical problems!

        # normalize the importance weights by their sum : omega[i,j] <- omega[i,j]/sum_j(omega[i,j])
        omegas_nrmlzd = torch.softmax(log_omegas, dim=1) # use the softmax to compute the exponential and
                                                         # normalize the omegas by their sum in a single step
        
        # reorder (useless)
        log_py_Ma = utils.reshape_fortran(log_py_Ma, (batch_size,K2))
        log_pM_Z  = utils.reshape_fortran(log_pM_Z,  (batch_size,K2))
        log_pZ    = utils.reshape_fortran(log_pZ,    (batch_size,K2))
        log_pa    = utils.reshape_fortran(log_pa,    (batch_size,K2))
        log_qZ_y  = utils.reshape_fortran(log_qZ_y,  (batch_size,K2))
        log_qM_Z  = utils.reshape_fortran(log_qM_Z,  (batch_size,K2))
        log_qa_My = utils.reshape_fortran(log_qa_My, (batch_size,K2))
        alphas_sup = utils.reshape_fortran(alphas_sup, (self.P,batch_size,K2))
        
        
        # now store the log probs of the spervised part
        log_probs_sup = {'log_py_Ma':log_py_Ma, 'log_pM_Z':log_pM_Z,
                         'log_pZ':log_pZ, 'log_qZ_y':log_qZ_y,
                         'log_qM_Z':log_qM_Z, 'log_qa_My':log_qa_My, 'log_pa':log_pa}
        
        # finally, store the alphas to regularize later
        alphas_all = {'alphas_unsup':alphas_unsup, 'alphas_sup':alphas_sup}
        
        return log_probs_unsup, log_probs_sup, omegas_nrmlzd, log_omegas, alphas_all
    




def my_loss_function(log_probs_unsup, log_probs_sup, omegas_nrmlzd, log_omegas, alphas_all):
    llambda = my_llambda; # regularization between supervised and unsupervised part
    bbeta   = 10; # extra regularization (high likelihood of endmembers and abundances training data in the posterior)
    tau     = my_tau; # extra extra regularization (sparsity)
    lamb_We = my_lamb_We; # penalizes network weights of nonlinear encoder
    lamb_Wd = my_lamb_Wd; # penalizes network weights of nonlinear decoder
    
    K1 = log_probs_unsup['log_py_Ma'].shape[1]
    K2 = omegas_nrmlzd.shape[1]
    batch_size = omegas_nrmlzd.shape[0]
    
    # unsupervised part of the cost function --------------
    cost_unsup = 0
    for i in range(0,batch_size):
        for j in range(0,K1):
            # terms in the numerator
            cost_unsup = cost_unsup + log_probs_unsup['log_py_Ma'][i,j] 
            cost_unsup = cost_unsup + log_probs_unsup['log_pa'][i,j]
            cost_unsup = cost_unsup + log_probs_unsup['log_pM_Z'][i,j] 
            cost_unsup = cost_unsup + log_probs_unsup['log_pZ'][i,j]
            # terms in the denominator
            cost_unsup = cost_unsup - log_probs_unsup['log_qa_My'][i,j]
            cost_unsup = cost_unsup - log_probs_unsup['log_qM_Z'][i,j]
            cost_unsup = cost_unsup - log_probs_unsup['log_qZ_y'][i,j]
            
    # supervised part of the cost function  
    cost_sup = 0
    for i in range(0,batch_size):
        for j in range(0,K2):
            temp = 0
            # terms in the numerator
            temp = temp + log_probs_sup['log_py_Ma'][i,j] 
            temp = temp + log_probs_sup['log_pa'][i,j] 
            temp = temp + log_probs_sup['log_pM_Z'][i,j] 
            temp = temp + log_probs_sup['log_pZ'][i,j] 
            # terms in the denominator
            temp = temp - log_probs_sup['log_qa_My'][i,j] 
            temp = temp - log_probs_sup['log_qM_Z'][i,j] 
            temp = temp - log_probs_sup['log_qZ_y'][i,j] 
            # importance weight normalization (the omegas are already normalized now)
            temp = omegas_nrmlzd[i,j] * temp
            # accumulate in the cost function
            cost_sup = cost_sup + temp
            
    # regularization term
    cost_reg = 0
    for i in range(0,batch_size):
        for j in range(0,K2):
            cost_reg = cost_reg + log_probs_sup['log_qa_My'][i,j] 
            cost_reg = cost_reg + log_omegas[i,j]
    
    # yet another regularization term (sparsity on alphas)
    cost_reg_sprs = 0
    for i in range(0,batch_size):
        # this one computes it over the unsupervised data
        for j in range(0,K1):
            cost_reg_sprs = cost_reg_sprs + torch.linalg.norm(alphas_all['alphas_unsup'][:,i,j], ord=0.5) / K1
        # this one computes it over the supervised data
        for j in range(0,K2):
            cost_reg_sprs = cost_reg_sprs + torch.linalg.norm(alphas_all['alphas_sup'][:,i,j], ord=0.5) / K2
    
    
    # regularizes nonlinear mixing weights
    reg_nlin_d_weights = 0
    for param in model.fcy_Ma_nlin.parameters():
        reg_nlin_d_weights = reg_nlin_d_weights + torch.norm(param, p="fro")
    
    reg_nlin_e_weights = 0
    for param in model.fca_My_alphas.parameters():
        reg_nlin_e_weights = reg_nlin_e_weights + torch.norm(param, p="fro")    
        
    
    # now the total cost functions
    cost = cost_unsup/K1 + llambda * cost_sup/K2 + (1+bbeta) * cost_reg/K2 \
           - tau * cost_reg_sprs - lamb_We * reg_nlin_e_weights - lamb_Wd * reg_nlin_d_weights
    return -cost # maximize cost





def train(epoch):
    log_interval = 10 # how many batches to wait before logging training status    
    model.train()
    train_loss = 0
    
    # get one batch from supervised data and from unsupervised data
    for batch_idx, alldata in enumerate(train_loader):
        optimizer.zero_grad()
        log_probs_unsup, log_probs_sup, omegas_nrmlzd, log_omegas, alphas_all = model(alldata)
        loss = my_loss_function(log_probs_unsup, log_probs_sup, omegas_nrmlzd, log_omegas, alphas_all)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(alldata), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(alldata)))
    
    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
    return avg_loss






def test(epoch):
    model.eval()
    with torch.no_grad():
        Y = torch.zeros((L,N))
        for i in range(N):
            Y[:,i] = train_loader.dataset.data_unsup[i]
        A_est, Mn_est, M_avg, Y_rec, a_nlin_deg = model.unmix(Y)
        
    return A_est, Mn_est, Y_rec, a_nlin_deg, M_avg







if __name__ == "__main__":

    
    # select example number and parameters    
    EX_NUM = 1
    
    
    if EX_NUM == 1: # DC1, synthetic data with nonlinear mixtures, BLMM
        my_llambda = 10
        my_tau     = 0
        my_lamb_We = 1e4
        my_lamb_Wd = 0.001
        
    if EX_NUM == 2: # DC2, synthetic data with endmember variability
        my_llambda = 1
        my_tau     = 0.005
        my_lamb_We = 0.01
        my_lamb_Wd = 0.1
        
    if EX_NUM == 3: # real Samson image
        my_llambda = 1
        my_tau     = 0.005
        my_lamb_We = 0.01
        my_lamb_Wd = 0.1
        
    if EX_NUM == 4: # real Jasper Ridge image
        my_llambda = 1
        my_tau     = 0
        my_lamb_We = 0.01
        my_lamb_Wd = 0.1
    
    if EX_NUM == 5: # real Cuprite image
        my_llambda = 10
        my_tau     = 0.1
        my_lamb_We = 0.05
        my_lamb_Wd = 5
        
        
    
    train_loader = torch.utils.data.DataLoader(DG.dataset_maker(data_opt=EX_NUM), batch_size=16, shuffle=True)

    L = train_loader.dataset.data_sup[0][1].shape[0]
    P = train_loader.dataset.data_sup[0][1].shape[1]
    N = len(train_loader.dataset.data_unsup)
    nr, nc = train_loader.dataset.A_u_cube.shape[0], train_loader.dataset.A_u_cube.shape[1]
    H = 2 # dimension of the latent EM space
    
    
    model = IDNet(P, L, H=H).to(device)    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    start_time = time.time()
    num_epochs = 30; # number of epochs to train
    loss_old = 1e30
    for epoch in range(1, num_epochs + 1):
        loss_t = train(epoch)
        A_est, Mn_est, Y_rec, a_nlin_deg, M_avg = test(epoch)
        
        # compute metrics -----------------------
        RMSE_A, NRMSE_A = utils.compute_metrics(train_loader.dataset.A_u, A_est)
        RMSE_M, NRMSE_M = utils.compute_metrics(train_loader.dataset.M_u_ppx, Mn_est)
        RMSE_Y, NRMSE_Y = utils.compute_metrics(train_loader.dataset.Y, Y_rec)
        
        metrics_str = '====> EPOCH: {:d}, Abundance NRMSE: {:.6f}, Endmember NRMSE: {:.6f}'.format(epoch, NRMSE_A, NRMSE_M)
        with open("results/metrics_ex" + str(train_loader.dataset.data_opt) + ".txt", "a") as text_file:
            print(metrics_str, file=text_file)
            print(metrics_str) # print to console too
        
        if epoch <= 10:
            scheduler.step() # reduce from 1e-3 to 1e-4 in 10 epochs with rate approx 0.8
        
        # check stopping condition
        if abs(loss_t - loss_old) / abs(loss_t) < 1e-2:
            break
        loss_old = loss_t
    elapsed_time = time.time()-start_time
    
    
    # plot abundances and average EMs ----------------------------------
    utils.plotAbunds(A_est, nr=nr, nc=nc, 
                     thetitle='learned abundances',
                     savepath='results/a_est_ex'+ str(train_loader.dataset.data_opt) +'.pdf')
    utils.plotEMs(M_avg, thetitle='learned avg EMs')
    
    # compare results to ground truth abundances if available
    utils.show_ground_truth(A_true=train_loader.dataset.A_u, Mgt_avg=train_loader.dataset.M_u_avg, nr=nr, nc=nc)
    
    
    print('====> FINAL: Abundance NRMSE: {:.6f}, Endmember NRMSE: {:.6f}'.format(NRMSE_A, NRMSE_M))
    print('====> Elapsed time: {:.6f}'.format(elapsed_time))
    # plt.figure(), plt.plot(Mn_est[:,0,0:20]), plt.show();
    
    # save .mat file with the results
    savemat('results/results_ex' + str(train_loader.dataset.data_opt) + '.mat', 
            {'A_est':A_est.numpy(),
             'Mn_est':Mn_est.numpy(),
             'A_true':train_loader.dataset.A_u.numpy(),
             'Mn_true':train_loader.dataset.M_u_ppx.numpy(),
             'Y_rec':Y_rec.numpy(),
             'a_nlin_deg':a_nlin_deg.numpy()})
    
    
    
    
    
    
    