import numpy as np
from numpy.random import multivariate_normal
def generate_gauss_classes(m,S,P,N):
    """
% FUNCTION
%   [X,y]=generate_gauss_classes(m,S,P,N)
% Generates a set of points that stem from c classes, given 
% the corresponding a priori class probabilities and assuming that each 
% class is modeled by a Gaussian distribution (also used in Chapter 2).
% 
% INPUT ARGUMENTS:
%   m:  lxc matrix, whose j-th column corresponds to the mean of 
%       the j-th class.
%   S:  lxlxc matrix. S(:,:,j) is the covariance matrix of the j-th normal 
%       distribution.
%   P:  c-dimensional vector whose j-th component is the a priori
%       probability of the j-th class.
%   N:  total number of data vectors to be generated.
%
% OUTPUT ARGUMENTS:
%   X:  lxN matrix, whose columns are the produced data vectors.
%   y:  N-dimensional vector whose i-th component contains the label
%       of the class where the i-th data vector belongs.
    
    """
    l,c=  m.shape
    X= np.array([multivariate_normal(mean=m[i,:],cov=S[i,:,:],size=int(P[i]*N)) for i in range(c)])
    y= np.ndarray.flatten(np.array([[i]*int((P[i]*N)) for i in range(c)]))
    return [X,y]