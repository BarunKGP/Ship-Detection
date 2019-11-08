Our implementation of the paper Ship Detection in Polarimetric SAR Images via Variational Bayesian Inference
# Express the polarimetric SAR image as a tensor, and decompose the SAR image as the sum of a sparse component associated with ships and a sea clutter component. These components are denoted by some latent variables. Then, introduce hierarchical priors of the latent variables to establish the probabilistic model of ship detection. By using variational Bayesian inference, estimate the posterior distributions of the latent variables. Finally, the ship detection result is obtained in the iterative Bayesian inference process. 

. First, we express the polarimetric SAR image as a tensor, and decompose the SAR image as the sum of a sparse component associated with ships and a sea clutter component. These components are denoted by some latent variables. Then, we introduce hierarchical priors of the latent variables to establish the probabilistic model of ship detection. By using
variational Bayesian inference, we estimate the posterior distributions of the latent variables. Finally, the ship detection result is
obtained in the iterative Bayesian inference process. By virtue of
the tensor representation of polarimetric SAR image, the proposed
approach explicitly uses all the polarization channels of the SAR
image, and avoids the possible information loss in scalar polarimetric feature representation. Moreover, the proposed approach
needs no sliding windows. The variational Bayesian inference process actually uses all the pixels instead of the limited pixels in sliding
windows. Thus, the proposed approach has good ship detection performance and shape preserving ability, which is especially suitable
for congested sea areas. Experimental results accomplished over
C-band RADARSAT-2 polarimetric SAR images demonstrate that
the proposed approach can achieve state-of-the-art ship detection
performance.
Index Terms—Polarimetry, ship d
