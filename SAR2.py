#%%
from PIL import Image
import numpy as np
import bayespy

import matplotlib.pyplot as plt

HH = Image.open('HH.tiff')
HV = Image.open('HV.tiff')

#%%
#HH.show()
#HV.show()
#HH.size

#%%
HV_np= np.array(HV)
HH_np= np.array(HH)
D=c=np.dstack((HH_np, HV_np))

D.shape
m,n,d=D.shape
#%%
alpha0=0.001       #hyperparameter
beta0=1-alpha0     #hyperparameter
E=bayespy.nodes.Beta([alpha0,beta0],plates=(m,n),name='E')

A_mat=np.zeros((m,n),dtype=int)
C_mat=np.zeros((m,n,2))

key=np.sort(np.reshape(D,-1))[int(m*n*0.9)]
print(key)

for i in range(m):
    for j in range(n):
        if D[i,j,0]>key and D[i,j,1]>key:
            A_mat[i,j]=1
        else:
            C_mat[i,j,:]=D[i,j,:]

plt.imshow(C_mat[:,:,0])
plt.show()
plt.imshow(A_mat)
plt.show()

#%%
N=m*n
D=2
K=5

from bayespy.nodes import Dirichlet, Categorical
alpha = Dirichlet(1e-5*np.ones(K),name='alpha')
Zi = Categorical(alpha,plates=(N,),name='zi')

from bayespy.nodes import Gaussian, Wishart
mui = Gaussian(np.zeros(D), 1e-5*np.identity(D),plates=(K,),name='mui')
Lambdai = Wishart(D, 1e-5*np.identity(D),plates=(K,),name='Lambdai')

from bayespy.nodes import Mixture
Y = Mixture(Zi, Gaussian, mui, Lambdai,name='Y')

Zi.initialize_from_random()

from bayespy.inference import VB
Q = VB(Y, mui, Lambdai, Zi, alpha)

Y.observe(np.reshape(C_mat,(-1,2)))

Q.update(repeat=10)
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
A.observe(A_mat)
# Initializations A,C and related paameters then initialization of all bayespy.nodes objects and mean_vec,precission_mat

#%%
Model=bayespy.inference.VB(A,E,mu,Lambda,C,PI)
#%%
Model.update(repeat=5)
#%%

#%%