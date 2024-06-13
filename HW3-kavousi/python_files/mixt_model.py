import numpy as np 
import pandas as pd
from numpy.random import multivariate_normal
def mixt_model(m,S,P,N):
    """
% FUNCTION
%   [X,y]=mixt_model(m,S,P,N,sed)
% Generates a set of data vectors that stem from a mixture of normal 
% distributions (also used in Chapter 2).
%
% INPUT ARGUMENTS:
%   m:  cxl matrix whose i-th row contains the
%       (l-dimensional) mean of the i-th normal distribution.
%   S:  cxlxl matrix whose i-th lxl two-dimensional "slice" is the
%       covariance matrix corresponding to the i-th normal distribution.
%   P:  c-dimensional vector whose i-th coordinate contains
%       the a priori probability for the i-th normal  distribution.
%   N:  the total number of points to be generated by the mixture
%       distribution.
%   sed:  the seed used for the initialization of the built-in MATLAB
%         random generator function "rand".
%
% OUTPUT ARGUMENTS:
%   X:  Nxl matrix whose columns are the produced vectors.
%   y:  N-dimensional vector whose i-th element indicates the
%       distribution generated the i-th vector.
    
    
    """
    c,l=  m.shape
    X=[None]*N
    y=[None]*N
    for i in range(N):
        j= (np.random.choice(c,1,p=P))[0]

        X[i]=(multivariate_normal(mean=m[j,:],cov=S[j,:,:]))
        y[i]=j
    X= np.array(X)
    y=np.array(y)
    #print(X.shape)
   # X= X.reshape(-1,3)
    #print(y.shape)
    return [X,y]
if __name__=="__main__":
    m1=np.array([0,0,0])
    m2=np.array([1,2,2])
    m3=np.array([3,3,4])
    m=np.array([m1,m2,m3])
    S1=np.identity(3) * 0.8
    S= np.array([S1,S1,S1])
    P=[1/3,1/3,1/3]
    X,y= mixt_model(m,S,P,N=100)
    X1,y1=  mixt_model(m,S,P,N=100)
   # print(X.shape)
    #print
    #print(X)
    #print(sum(y==2))