from comp_gauss_dens_val import comp_gauss_dens_val
import numpy as np
def bayes_classifier(m,S,P,X):
    """
    % INPUT ARGUMENTS:

    m:      cxl matrix, whose j-th column is the mean of the j-th class.
    S:      cxlxl matrix, where S(j,:,:) corresponds to
            the covariance matrix of the normal distribution of the j-th
            class.
    P:      c-dimensional vector, whose j-th component is the a priori
            probability of the j-th class.
    X:      Nxl matrix, whose columns are the data vectors to be
            classified.
OUTPUT ARGUMENTS:
   z:      N-dimensional vector, whose i-th element is the label
           of the class where the i-th data vector is classified.

    
    """
    c,l=m.shape
    N,l=X.shape
    print(N,c)
    z=[None]*N
    for i in range(N):
        
        P_w_x=[None]*c
        for j in range(c):
            P_w_x[j]= P[j]*comp_gauss_dens_val(m[j,:],S[j,:,:],X[i,:])
        z[i]=  np.argmax(P_w_x)
    return z
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
    print(bayes_classifier(m,S,P,x))