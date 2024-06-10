import numpy as np
import math
def comp_gauss_dens_val(m,S,x):
    """
% FUNCTION
%   [z]=comp_gauss_dens_val(m,S,x)
% Computes the value of a Gaussian distribution, N(m,S), at a specific point 
% (also used in Chapter 2).
%
% INPUT ARGUMENTS:
%   m:  l-dimensional column vector corresponding to the mean vector of the
%       gaussian distribution.
%   S:  lxl matrix that corresponds to the covariance matrix of the 
%       gaussian distribution.
%   x:  l-dimensional column vector where the value of the gaussian
%       distribution will be evaluated.
%
% OUTPUT ARGUMENTS:
%   z:  the value of the gaussian distribution at x.
    
    """
    l= m.shape[0]
    z = ((1/(2*np.pi)**(l/2))*(np.linalg.det(S)**0.5))*math.exp(-0.5*(x-m).T@np.linalg.inv(S)@(x-m))
    return np.array(z)
if __name__=="__main__":
    m=np.array([0,1])
    S=np.array([[1,0],[0,1]])
    x1=np.array([0.2,1.3])
    print(comp_gauss_dens_val(m,S,x1))