import numpy as np
def euclidean_classifier(m,X):
    """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%   [z]=euclidean_classifier(m,X)
% Euclidean classifier for the case of c classes.
%
% INPUT ARGUMENTS:
%   m:  cxl matrix, whose i-th column corresponds to the mean of the i-th
%       class.
%   X:  Nxl matrix whose columns are the data vectors to be classified.
%
% OUTPUT ARGUMENTS:
%   z:  N-dimensional vector whose i-th element contains the label
%       of the class where the i-th data vector has been assigned.
%
    """
    c,l=m.shape
    N,l=X.shape
    z=[None]*N
    d_e_mat=np.zeros(shape=(N,c))
    for i in range(N):
        
        d_e=[None]*c
        for j in range(c):
            d_e[j]=np.sqrt((X[i,:]-m[j,:]).T@(X[i,:]-m[j,:]))
            d_e_mat[i,j]=d_e[j]
        z[i]=  np.argmin(d_e)
        
    return z,d_e_mat
if __name__=="__main__":
    m1=np.array([0,0,0])
    m2=np.array([0.5,0.5,0.5])
    m=np.array([m1,m2])
    S=np.array([[0.8,0.01,0.01],
                [0.01,0.2,0.01],
                [0.01,0.01,0.2]])
    S=np.array([S,S])
    x=np.array([[0.1,0.5,0.1]])
    P=[0.5,0.5]
    print(euclidean_classifier(m,x))