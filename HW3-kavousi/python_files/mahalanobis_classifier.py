import numpy as np
def mahalanobis_classifier(m,S,X):
    """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%   [z]=mahalanobis_classifier(m,S,X)
% Mahalanobis classifier for c classes.
%
% INPUT ARGUMENTS:
%   m:  cxl matrix, whose i-th column corresponds to the
%       mean of the i-th class
%   S:  cxlxl matrix which corresponds to the matrix
%       involved in the Mahalanobis distance (when the classes have
%       the same covariance matrix, S equals to this common covariance
%       matrix).
%   X:  Nxl matrix, whose columns are the data vectors to be classified.
%
% OUTPUT ARGUMENTS:
%   z:  N-dimensional vector whose i-th component contains the label
%       of the class where the i-th data vector has been assigned.
%   d_m_mat: matrix showing eulidean distance values . d_e is is Nxl. d_e_mat[i,j]-> i,j where ith sample and jth class    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    
    """
    c,l=m.shape
    N,l=X.shape
    z=[None]*N
    d_m_mat=np.zeros(shape=(N,c))
    for i in range(N):
        
        d_m=[None]*c
        for j in range(c):
            d_m[j]=np.sqrt((X[i,:]-m[j,:]).T@(np.linalg.inv(S[j,:,:]))@(X[i,:]-m[j,:]))
            #print(d_m[j])
            d_m_mat[i,j]=d_m[j]
        z[i]=  np.argmin(d_m)
        
    return np.array(z),d_m_mat
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
    print(mahalanobis_classifier(m,S,x))