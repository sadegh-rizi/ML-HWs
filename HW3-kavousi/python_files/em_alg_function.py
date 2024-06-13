import numpy as np 
import pandas as pd
from comp_gauss_dens_val import comp_gauss_dens_val
import random

random.seed(0)


def em_alg_function(x,m,s,Pa,e_min):
    """
% FUNCTION
%   [m,s,Pa,iter,Q_tot,e_tot]=em_alg_function(x,m,s,Pa,e_min)
% EM algorithm for estimating the parameters of a mixture of normal
% distributions, with diagonal covariance matrices.
% WARNING: IT ONLY RUNS FOR THE CASE WHERE THE COVARIANCE MATRICES
% ARE OF THE FORM sigma^2*I. IN ADDITION, IF sigma_i^2=0 FOR SOME
% DISTRIBUTION AT AN ITERATION, IT IS ARBITRARILY SET EQUAL TO 0.001.
%
% INPUT ARGUMENTS:
%   x:      Nxl matrix, each row of which is a feature vector.
%   m:      Jxl matrix, whos j-th row is the initial
%           estimate for the mean of the j-th distribution.
%   s:      J*1 vector, whose j-th element is the variance
%           for the j-th distribution.
%   Pa:     J-dimensional vector, whose j-th element is the initial
%           estimate of the a priori probability of the j-th distribution.
%   e_min:  threshold used in the termination condition of the EM
%           algorithm.
%
% OUTPUT ARGUMENTS:
%   m:      it has the same structure with input argument m and contains
%           the final estimates of the means of the normal distributions.
%   s:      it has the same structure with input argument s and contains
%           the final estimates of the variances of the normal
%           distributions.
%   Pa:     J-dimensional vector, whose j-th element is the final estimate
%           of the a priori probability of the j-th distribution.
%   iter:   the number of iterations required for the convergence of the
%           EM algorithm.
%   Q_tot:  vector containing the likelihood value at each iteration.
%   e_tot:  vector containing the error value at each itertion.
%
    
    
    """
    N,l=x.shape
    J=m.shape[0]
    Q_tot= np.array([])
    e_tot=np.array([])
    e=np.inf
    P_jk= np.zeros(shape=(J,N))
    iter=0
    while e>e_min:
        P_old=Pa.copy()
        m_old=m.copy()
        s_old=s.copy()
        for k in range(N):
            temp = np.array([comp_gauss_dens_val(m[j,:],S[j,:],x[k,:]) for j in range(J)])
            P_tot = temp@Pa
            for j in range(J):
                
                P_jk[j,k] = temp[j]*Pa[j]/P_tot

        # for k in range(N):
        #     P_tot = 0
        #     for j in range(J):
        #         temp = comp_gauss_dens_val(m[j,:],S[j,:],x[k,:])

        #         P_jk[j,k] = temp*Pa[j]
        #         P_tot+=P_jk[j,k]
        #     P_jk=P_jk/P_tot
            
        #Determine the log-likelihood
        Q=0
        for k in range(N):
            for j in range(J):
                Q+=  P_jk[j,k]*(-(l/2)*np.log(2*np.pi*s[j])-(1/2*s[j])*(np.linalg.norm(x[k,:]-m[j,:])**2*+np.log(Pa[j])))

        Q_tot =  np.append(Q_tot,Q)
        #determine the means
        for j in range(J):
            a=np.zeros(shape=(1,l))
            #a is the numerator
            for k in range(N):
                a+=P_jk[j,k]*x[k,:]
                #print(a)
            #print(a)
            #print(P_jk[j,:].shape)
            #print((sum(P_jk[j,:])))
            #print(a.shape)
            #print((a/sum(P_jk[j,:])).flatten())
            #print(m[j,:].shape)
            #print(((a/sum(P_jk[j,:])).flatten()))
            m[j,:]=(a/sum(P_jk[j,:])).flatten()
            #print(m[j,:])

            #print(m[j,:])
        #determine the variances
        for j in range(J):
            b=0
            #a is the numerator
            
            for k in range(N):
                b+= P_jk[j,k]*(((x[k,:]-m[j,:]))@((x[k,:]-m[j,:]).T))
            # print(b/(l*sum(P_jk[j,:])))
            s[j]=b/(l*sum(P_jk[j,:]))
            # print(s[j])


        # if s[j]<1e-10:
        #     s[j]=0.001

        #determine a priori probability
        for j in range(J):
            a=0
            for k in range(1,N):
                a+=P_jk[j,k]
            Pa[j]=a/N
        # print(m)
        # print(m_old)
        # print(abs(m-m_old))
        # print((sum(abs(m-m_old))))


        e = sum(abs(Pa-P_old)+sum(sum(abs(m-m_old)))+sum(abs(s-s_old)))
        #e = (np.linalg.norm(Pa-P_old)+(np.linalg.norm(abs(m-m_old)))+(np.linalg.norm(s-s_old)))

        # print(Pa,P_old)
        # print(abs(Pa-P_old))
        # print(e)
        #print(P_old)

        #print(Pa)
        e_tot=np.append(e_tot,e)
        #print(e)
        #print(e_min)
        iter=iter+1

    return [m,s,Pa,iter,Q_tot,e_tot]
if __name__=="__main__":
    from mixt_model import mixt_model  
    m=np.zeros(shape=[3,2],dtype=float)  
    
    m[0,:]=np.array([1,1])
    m[1,:]=np.array([3,3])
    m[2,:]=np.array([2,6])
    S=np.zeros(shape=(3,2,2),dtype=float)
    S[0,:,:]=0.1*np.identity(2)
    S[1,:,:]=0.2*np.identity(2)
    S[2,:,:]=0.3*np.identity(2)

    P=np.array([0.4,0.4,0.2])
    N=500
    X,y=mixt_model(m,S,P,N)
    #print(X.shape)
    m1_ini=np.array([0,2])
    m2_ini=np.array([5,2])
    m3_ini=np.array([5,5])
    m_ini=np.array([m1_ini,m2_ini,m3_ini],dtype=float)
    e_min=1e-5
    s_ini=np.array([0.15,0.27,0.4])
    Pa_ini=np.array([1/3,1/3,1/3])   
    print("Outputtt")
    print(em_alg_function(X,m_ini,s_ini,Pa_ini,e_min))

    m_hat,s_hat,Pa,iter,Q_tot,e_tot=em_alg_function(X,m_ini,s_ini,Pa_ini,e_min)
    # print(m_hat)