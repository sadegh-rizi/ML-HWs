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
%   m:  cxl matrix, whose j-th column corresponds to the mean of 
%       the j-th class.
%   S:  cxlxl matrix. S(:,:,j) is the covariance matrix of the j-th normal 
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
    X= X.reshape(-1,3)
    y= np.ndarray.flatten(np.array([[i]*int((P[i]*N)) for i in range(c)]))
    return [X,y]
if __name__=="__main__":
    m1=np.array([0,0,0])
    m2=np.array([1,2,2])
    m3=np.array([3,3,4])
    m=np.array([m1,m2,m3])
    S1=np.identity(3) * 0.8
    S= np.array([S1,S1,S1])
    P=[1/3,1/3,1/3]
    X,y= generate_gauss_classes(m,S,P,N=1000)
    X1,y1=  generate_gauss_classes(m,S,P,N=1000)
    print(X.shape)
    print