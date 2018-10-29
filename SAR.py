# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 21:55:54 2018

@author: KTJ
"""

#%%
from PIL import Image
import numpy as np
import bayespy

HH = Image.open('HH.tiff')
HV = Image.open('HV.tiff')
#%%
HH.show()
#%%
HV.show()
#%%
# import tifffile as tiff

#%%
#a = tiff.imread('HH.tiff')
#a.shape
HH.size
#%%
#%%
HV_np= np.array(HV)
HH_np= np.array(HH)
D=c=np.dstack((HH_np, HV_np))
D.shape
#%%
m,n,d=D.shape
#%%
alpha0=0.001       #hyperparameter
beta0=1-alpha0     #hyperparameter
E=bayespy.nodes.Beta([alpha0,beta0],plates=(m,n),name='E')
#%%
K=5                     #hyperparameter
neta=1e-6*np.ones(K)    #hyperparameter
print(neta.shape)
print(neta)
PI=bayespy.nodes.Dirichlet(neta,name='PI')  
#%%
Z=bayespy.nodes.Categorical(PI,plates=(m,n,K),name='Z')

mean_vec=np.zeros(d)  # to be initialized accorinding to image
precission_mat=1e-5*np.identity(d)     # to be initialized accorinding to image
mu=bayespy.nodes.Gaussian(mean_vec,precission_mat,plates=(K,),name='U')

Lambda=bayespy.nodes.Wishart(d,precission_mat,plates=(K,),name='Lambda')

C=bayespy.nodes.Mixture(Z,bayespy.nodes.Gaussian,mu,Lambda,name='C')
#%%
A=bayespy.nodes.Bernoulli(E,plates=(m,n),name='A')
#%%
# Initializations A,C and related paameters then initialization of all bayespy.nodes objects and mean_vec,precission_mat

#%%
Model=bayespy.inference.VB(A,E,mu,Lambda,C,PI)
#%%
Model.update(repeat=5)
#%%

#%%