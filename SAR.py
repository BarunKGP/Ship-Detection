from PIL import Image
import bayespy

HH = Image.open('HH.tiff')
HH.show() # opens the tiff image. this rainbow color tiff
HV = Image.open('HV.tiff')
HV.show() # opens the tiff image. this rainbow color tiff

import numpy as np
HV = np.array(HV)
HH = np.array(HH)
print(np.shape(HV))
print(np.max(HH),np.max(HV))

m=np.shape(HH)[0]
n=np.shape(HH)[1]

D=np.zeros((m,n,2))  #Image Tensor
D[:,:,0]=HV
D[:,:,1]=HH

K = 5 #No. of clusters

#Initialization of Latent Variables
A=np.zeros((m,n))
E=np.zeros((m,n))
U=np.zeros((m,n))
M=np.zeros((m,n))
Z=np.zeros((m,n))
pi=np.zeros((m,n))
phi=np.zeros((m,n))

from bayespy import nodes
mu = nodes.Gaussian ( np . zeros (D) , 0.01* np . identity ( D ) , plates =( K ,) )
Lambda = nodes . Wishart (D , D * np . identity ( D ) , plates =( K ,) )

alpha = nodes . Dirichlet (0.01* np . ones ( K ) )
Z = nodes . Categorical ( alpha , plates =(M,))

y = nodes . Mixture (Z, nodes.Gaussian , mu , Lambda )
y . observe (D)

from bayespy . inference import VB
Q = VB (y , mu , Z , Lambda , alpha )

Z . initialize_from_random ()
Q . update ( repeat =10) 

import bayespy . plot as bpplt
bpplt . gaussian_mixture_2d (y , alpha = alpha )
bpplt . pyplot . show ()
print ( alpha )

def Update(A):
    return A

def Calculate(A):
    return A

for i in range(10):
    A=Update(A)
    E=Update(E)
    U=Update(U)
    M=Update(M)
    Z=Update(Z)
    pi=Update(pi)
    
Calculate(A)
