#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 21:49:06 2021

# loads hyperspectral images

@author: ricardo
"""

import torch
import torch.distributions as td
import scipy
import numpy as np
from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt
import utils



# paths to the dataset files: -------------------------------------------------
  
path_dataset_DC1 = ['DATA/synth_DC1/data_ex_nl1.mat',\
                    'DATA/synth_DC1/extracted_bundles_nl_ex1.mat'] # synthetic data with nonlinear mixtures, with the BLMM

path_dataset_DC2 = ['DATA/synth_DC2/alldata_ex_DC2.mat',\
                    'DATA/synth_DC2/extracted_bundles.mat'] # synthetic data with spectral variability

path_dataset_Samson = ['DATA/real_Samson/alldata_real_Samson.mat',\
                       'DATA/real_Samson/extracted_bundles.mat']

path_dataset_Jasper = ['DATA/real_Jasper/alldata_real_Jasper.mat',\
                       'DATA/real_Jasper/extracted_bundles.mat']

path_dataset_Cuprite = ['DATA/real_Cuprite/alldata_real_Cuprite.mat',\
                        'DATA/real_Cuprite/extracted_bundles.mat']




class dataNonlinear_synth():
    def __init__(self):
        '''data from synthetic example with nonlinear mixtures (BLMM)'''
        mat_contents1 = loadmat(path_dataset_DC1[0]) # load image
        mat_contents2 = loadmat(path_dataset_DC1[1]) # load image-extracted spectral libraries

        self.L = mat_contents1['Mth'].shape[0]
        self.P = mat_contents1['Mth'].shape[1]
        self.N = mat_contents1['Y'].shape[1]
        self.Nlib = mat_contents2['bundleLibs'][0,0].shape[1]
        self.Y = torch.from_numpy(mat_contents1['Y']).type(torch.float32)
        
        SNR = 40 
        ssigma = (self.Y.mean())*(10**(-SNR/10))
        
        self.data_sup = []
        for i in range(self.Nlib):
            M_s = torch.zeros((self.L,self.P))
            for j in range(self.P):
                m_ij = mat_contents2['bundleLibs'][0,j][:,i]                
                M_s[:,j] = torch.from_numpy(m_ij)            
            for k in range(self.P):
                a_s = torch.zeros((self.P,))
                a_s[k] = 1.0
                y_s = torch.mv(M_s,a_s) + ssigma * torch.randn((self.L,))
                self.data_sup.append((y_s,M_s,a_s))
                
        self.data_unsup = []
        for i in range(self.N):
            self.data_unsup.append(torch.from_numpy(mat_contents1['Y'][:,i]).type(torch.float32))
        self.A_cube_gt = torch.from_numpy(mat_contents1['A_cube'])
        self.A_gt = self.A_cube_gt.permute(1,0,2).reshape((self.N,self.P)).T
        self.Mavg_th = torch.from_numpy(mat_contents1['Mth'])
        self.Mn_th = -0.5*torch.ones(self.L,self.P,self.N) 
        
    def getdata(self):
        return self.Y, self.data_sup, self.data_unsup
    
    def get_gt(self):
        return self.A_gt, self.A_cube_gt, self.Mavg_th, self.Mn_th

        


        

class dataVariability_synth():
    def __init__(self):
        '''data from synthetic example with spectral variability'''
        mat_contents1 = loadmat(path_dataset_DC2[0]) # load image
        mat_contents2 = loadmat(path_dataset_DC2[1]) # load image-extracted spectral libraries

        
        self.L = mat_contents1['Mth'].shape[0]
        self.P = mat_contents1['Mth'].shape[1]
        self.N = mat_contents1['Mth'].shape[2]
        self.Nlib = mat_contents2['bundleLibs'][0,0].shape[1]
        self.Y = torch.from_numpy(mat_contents1['Y']).type(torch.float32)
        
        SNR = 40 
        ssigma = (self.Y.mean())*(10**(-SNR/10))
        
        self.data_sup = []
        for i in range(self.Nlib):
            M_s = torch.zeros((self.L,self.P))
            for j in range(self.P):
                m_ij = mat_contents2['bundleLibs'][0,j][:,i]                
                M_s[:,j] = torch.from_numpy(m_ij)            
            for k in range(self.P):
                a_s = torch.zeros((self.P,))
                a_s[k] = 1.0
                y_s = torch.mv(M_s,a_s) + ssigma * torch.randn((self.L,))
                self.data_sup.append((y_s,M_s,a_s))
                
        self.data_unsup = []
        for i in range(self.N):
            self.data_unsup.append(torch.from_numpy(mat_contents1['Y'][:,i]).type(torch.float32))
        self.A_gt = torch.from_numpy(mat_contents1['A'])
        self.A_cube_gt = torch.from_numpy(mat_contents1['A_cube'])
        self.Mavg_th = torch.from_numpy(mat_contents1['M_avg'])
        self.Mn_th = torch.from_numpy(mat_contents1['Mth'])
        
    def getdata(self):
        return self.Y, self.data_sup, self.data_unsup
    
    def get_gt(self):
        return self.A_gt, self.A_cube_gt, self.Mavg_th, self.Mn_th




class dataReal_real():
    def __init__(self, ex_num=1):
        '''data from real examples
        ex_num \in \{1,2,3\} = \{Samson, Jasper Ridge, Cuprite\}'''
        
        if ex_num == 1: # load data for the Samson image
            mat_contents1 = loadmat(path_dataset_Samson[0]) # load image
            mat_contents2 = loadmat(path_dataset_Samson[1]) # load image-extracted spectral libraries
        
        if ex_num == 2: # load data for the Jasper Ridge image
            mat_contents1 = loadmat(path_dataset_Jasper[0]) # load image
            mat_contents2 = loadmat(path_dataset_Jasper[1]) # load image-extracted spectral libraries
        
        if ex_num == 3: # load data for the Cuprite image
            mat_contents1 = loadmat(path_dataset_Cuprite[0]) # load image
            mat_contents2 = loadmat(path_dataset_Cuprite[1]) # load image-extracted spectral libraries

            
        self.L = mat_contents1['M0'].shape[0]
        self.P = mat_contents1['M0'].shape[1]
        self.N = mat_contents1['Y'].shape[1]
        self.Nlib = mat_contents2['bundleLibs'][0,0].shape[1]
        self.nr, self.nc = mat_contents1['Yim'].shape[0], mat_contents1['Yim'].shape[1]
        self.Y = torch.from_numpy(mat_contents1['Y']).type(torch.float32)
        
        SNR = 40 
        ssigma = (self.Y.mean())*(10**(-SNR/10))
        
        self.data_sup = []
        for i in range(self.Nlib):
            M_s = torch.zeros((self.L,self.P))
            for j in range(self.P):
                # m_ij = mat_contents2['bundleLibs'][0,j][:,i]  
                m_ij = mat_contents2['bundleLibs'][0,j][:,i%mat_contents2['bundleLibs'][0,j].shape[1]] # circular shift
                # m_ij = mat_contents1['M0'][:,j]
                M_s[:,j] = torch.from_numpy(m_ij)
            for k in range(self.P):
                a_s = torch.zeros((self.P,))
                a_s[k] = 1.0
                y_s = torch.mv(M_s,a_s) + ssigma * torch.randn((self.L,))
                self.data_sup.append((y_s,M_s,a_s))
                
        self.data_unsup = []
        for i in range(self.N):
            self.data_unsup.append(torch.from_numpy(mat_contents1['Y'][:,i]).type(torch.float32))
        self.A_gt = -torch.ones((self.P,self.N))
        self.A_cube_gt = -torch.ones((self.nr,self.nc,self.P))
        self.Mavg_th = torch.from_numpy(mat_contents1['M0'])
        self.Mn_th = -torch.ones((self.L,self.P,self.N))
        
    def getdata(self):
        return self.Y, self.data_sup, self.data_unsup
    
    def get_gt(self):
        return self.A_gt, self.A_cube_gt, self.Mavg_th, self.Mn_th
    
    
    




class dataset_maker(torch.utils.data.Dataset):
    def __init__(self, data_opt = 1):
        '''initialize variables and select which data to load
        data_opt = 1 : synthetic nonlinear mixture (DC1, with the BLMM) 
        data_opt = 2 : synthetic example with variability (DC2)
        data_opt = 3--5 : real data examples (samson, jasper, cuprite)
        '''
        self.data_sup = []
        self.data_unsup = []
        # the following is the data ground truth:
        self.A_u = [] # ground truth abundance matrix (P * N)
        self.A_u_cube = [] # ground truth abundance cube (nr * nc * P) 
        self.M_u_avg = [] # ground truth 'average' or 'reference' EM matrix
        self.M_u_ppx = [] # ground truth EM matrices for each pixel
        self.Y = [] # observed hyperspectral image (L * N)
        
        self.data_opt = data_opt # store data index for later access
        if data_opt == 1:
            self.NonlinearData_synth()
        if data_opt == 2: 
            self.VariabilityData_synth()
        if data_opt in range(3,6):
            self.RealData_real(data_opt-2)
        
        if len(self.data_sup) < len(self.data_unsup):
            self.flag_unsup_is_bigger = True
        else:
            self.flag_unsup_is_bigger = False
            
            

    def NonlinearData_synth(self):
        myDatator = dataNonlinear_synth()
        self.Y, self.data_sup, self.data_unsup = myDatator.getdata()
        self.A_u, self.A_u_cube, self.M_u_avg, self.M_u_ppx = myDatator.get_gt()
    
    def VariabilityData_synth(self):
        myDatator = dataVariability_synth()
        self.Y, self.data_sup, self.data_unsup = myDatator.getdata()
        self.A_u, self.A_u_cube, self.M_u_avg, self.M_u_ppx = myDatator.get_gt()
    
    def RealData_real(self, k):
        '''k \in {1,2,3}'''
        myDatator = dataReal_real(k)
        self.Y, self.data_sup, self.data_unsup = myDatator.getdata()
        self.A_u, self.A_u_cube, self.M_u_avg, self.M_u_ppx = myDatator.get_gt()
        
        
    def __len__(self):
        # take the maximum length between the supervised and unsupervised datasets
        return max(len(self.data_sup),len(self.data_unsup))

    def __getitem__(self, idx):
        # now, idx corresponds to the index among the largest (sup or unsup) dataset.
        # We can multiply it by the ratio between the smallest and the largest datset
        # and round down to an integer, to obtain the corresponding index for the 
        # smaller dataset
        if self.flag_unsup_is_bigger:
            idx_sup = int(np.floor(idx*len(self.data_sup)/len(self.data_unsup)))
            idx_unsup = idx
        else:
            idx_sup = idx
            idx_unsup = int(np.floor(idx*len(self.data_unsup)/len(self.data_sup)))
        
        # return tuples? (y) and (y,M,a)
        return self.data_unsup[idx_unsup], self.data_sup[idx_sup]
    
    
    def plot_training_EMs(self, EM_idx=-1):
        '''small method to plot EMs in the training dataset'''
        L, P, Nsamp = self.data_sup[0][1].shape[0], self.data_sup[0][1].shape[1], len(self.data_sup)
        M_train = torch.zeros((L,P,Nsamp))
        for i in range(Nsamp):
            M_train[:,:,i] = self.data_sup[0][1]
        if EM_idx == -1:
            fig, axs = plt.subplots(1, P)
            for i in range(P):
                axs[i].plot(torch.squeeze(M_train[:,i,:]))
        else:
            plt.figure()
            plt.plot(torch.squeeze(M_train[:,EM_idx,:]))
            plt.show()
                




if __name__ == "__main__":
    datator = dataVariability_synth()
    Y, M, A = datator.samplePixels_LMM(10,2,10,2,30)
    datator2 = dataset_maker(data_opt = 5-3)
    datator2.__getitem__(20)
    
    
    
    