import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
from itertools import product

def Z_generator(N: list, n: list, replication: int, gen: callable=np.random) -> np.matrix:
    k=len(N)
    Z_perm=[[]]*k
    for i in range(k):
        Z_perm[i]=np.block([np.ones((replication,n[i])),np.zeros((replication,N[i]-n[i]))])
    shuffle=gen.shuffle
    for i in range(k):
        for j in range(replication):
            shuffle(Z_perm[i][j])
    return Z_perm


def filter_table(matrix_ni: list, matrix_Ni: list) -> bool:
    return max(0,matrix_ni[0]-matrix_Ni[1], matrix_Ni[0]-matrix_ni[2], matrix_Ni[0]+matrix_Ni[2]-matrix_ni[1]-matrix_ni[2]) <= min(matrix_Ni[0], matrix_ni[0], matrix_Ni[0]+matrix_Ni[2]-matrix_ni[2], np.sum(matrix_Ni)-matrix_Ni[1]-matrix_ni[1]-matrix_ni[2])



def default_T(matrix_n: list, matrix_N: list) -> float:
    k=len(matrix_N)
    N=[np.sum(matrix_N[i]) for i in range(k)]
    n=[matrix_n[i][0]+matrix_n[i][1] for i in range(k)]
    tau_hat=sum(N[i]/np.sum(N)*((matrix_N[i][1]-matrix_N[i][2])/N[i]-matrix_n[i][0]/n[i]+matrix_n[i][2]/(N[i]-n[i])) for i in range(k))
    return round(abs(tau_hat),10)

def p_value_matrix(matrix_n: list, matrix_N: list, Z_perm: list) -> float:
    k=len(matrix_N)
    N=[np.sum(matrix_N[i]) for i in range(k)]
    n=[matrix_n[i][0]+matrix_n[i][1] for i in range(k)]
    replication=np.shape(Z_perm[0])[0]
    tau=0
    tau_hat=np.zeros(replication)
    for i in range(k):
        N11,N10,N01,N00=matrix_N[i]

        dat=np.zeros((N[i],2))
        dat[:N11,:]=1
        dat[N11:N11+N10,0]=1
        dat[N11:N11+N10,1]=0
        dat[N11+N10:N11+N10+N01,0]=0
        dat[N11+N10:N11+N10+N01,1]=1
        dat[N11+N10+N01:,:]=0
        tau_hat+=N[i]/np.sum(N)*(np.dot(Z_perm[i],dat[:,0])/n[i]-np.dot((1-Z_perm[i]),dat[:,1])/(N[i]-n[i]))
        tau+=(N10-N01)/np.sum(N)
    p=sum(np.round(abs(tau-tau_hat),10)>=default_T(matrix_n,matrix_N))/replication
    return p



def Ni_generator(matrix_ni: list) -> tuple:
    n11,n10,n01,n00=matrix_ni
    N=sum(matrix_ni)
    res=[]
    for i in range(min(N-n00, N-n10)+1):
        N11 = i
        for j in range(max(0, n01-N11), N-n00-N11+1):
            N01 = j
            for k in range(max(0, n11-N11), min(N-n10-N11, N-N11-N01)+1):
                N10 = k
                N00 = N-N11-N10-N01
                if filter_table(matrix_ni, [N11,N10,N01,N00]):
                    res.append([N11,N10,N01,N00])
    return res

def Ni_last_two_generator(matrix_ni: list) -> tuple:
    n11,n10,n01,n00=matrix_ni
    N=sum(matrix_ni)
    res=[]
    for i in range(min(N-n00, N-n10)+1):
        N11 = i
        for j in range(max(0, n01-N11), N-n00-N11+1):
            N01 = j
            for k in range(max(0, n11-N11), min(N-n10-N11, N-N11-N01)+1):
                N10 = k
                N00 = N-N11-N10-N01
                if filter_table(matrix_ni, [N11,N10,N01,N00]):
                    res.append(tuple([0,0,N01,N00]))
    return list(set(res))

def confidence_interval_permute(matrix_n,alpha=0.05,replication=10000):
    #Suppose n^1<=N^1/2, n^2<N^2/2
    k=len(matrix_n)
    N=[np.sum(matrix_n[i]) for i in range(k)]
    n=[matrix_n[i][0]+matrix_n[i][1] for i in range(k)]
    tauhat=sum(N[i]/sum(N)*(matrix_n[i][0]/n[i]-matrix_n[i][2]/(N[i]-n[i])) for i in range(k))
    L=np.sum(matrix_n)
    U=-np.sum(matrix_n)
    Z_perm=Z_generator(N, n, replication)
    pre_matrix_N_list=[]

    for i in range(2):
        temp=Ni_last_two_generator(matrix_n[i])
        pre_matrix_N_list.append(temp)

    for i in range(2,k):
        temp=Ni_generator(matrix_n[i])
        pre_matrix_N_list.append(temp)

    num=0

    #find the upper bound
    for pre_matrix_N in product(*pre_matrix_N_list):
        a,b,M101,M100=pre_matrix_N[0]
        a,b,M201,M200=pre_matrix_N[1]
        N110=N[0]-M101-M100
        for N210 in range(N[1]-M201-M200+1):
            if filter_table(matrix_n[1],[N[1]-N210-M201-M200,N210,M201,M200])==False:
                continue
            N110low=math.floor(tauhat*sum(N))+M101+M201-N210-sum(pre_matrix_N[i][1]-pre_matrix_N[i][2] for i in range(2,k))
            while True:
                matrix_N=list(pre_matrix_N)
                matrix_N[0]=[N[0]-N110-M101-M100,N110,M101,M100]
                matrix_N[1]=[N[1]-N210-M201-M200,N210,M201,M200]
                tau=sum(matrix_N[i][1]-matrix_N[i][2] for i in range(k))/sum(N)
                if tau<tauhat:
                    N110up=N110
                    break
                p_value=p_value_matrix(matrix_n,matrix_N,Z_perm)
                num+=1
                if p_value>alpha:
                    N110up=N110
                    break
                N110-=1
            
            #find the minimum and maximum value of N^1_{10} such that matrix_N is compatible with the data
            tempa=max(matrix_n[0][0],N[0]-M101-M100-matrix_n[0][2],N[0]-M100-matrix_n[0][1]-matrix_n[0][2])
            tempb=min(N[0]-M101-M100,N[0]-M100-matrix_n[0][2],N[0]-matrix_n[0][1]-matrix_n[0][2])
            N110low2=tempa-matrix_n[0][0]
            N110up2=tempb
            #algorithm 4d
            max_N110=min(N110up2,N110up)
            if max_N110>=N110low2:
                tau=max_N110-M101+N210-M201+sum(pre_matrix_N[i][1]-matrix_N[i][2] for i in range(2,k))
                if tau>U:
                    U=tau

    #find the lower bound
    for pre_matrix_N in product(*pre_matrix_N_list):
        a,b,M101,M100=pre_matrix_N[0]
        a,b,M201,M200=pre_matrix_N[1]
        N110=0
        for N210 in range(N[1]-M201-M200,-1,-1):
            if filter_table(matrix_n[1],[N[1]-N210-M201-M200,N210,M201,M200])==False:
                continue
            N110up=math.ceil(tauhat*sum(N))+M101+M201-N210-sum(pre_matrix_N[i][1]-pre_matrix_N[i][2] for i in range(2,k))
            while True:
                matrix_N=list(pre_matrix_N)
                matrix_N[0]=[N[0]-N110-M101-M100,N110,M101,M100]
                matrix_N[1]=[N[1]-N210-M201-M200,N210,M201,M200]
                tau=sum(matrix_N[i][1]-matrix_N[i][2] for i in range(k))/sum(N)
                if tau>tauhat:
                    N110low=N110
                    break
                p_value=p_value_matrix(matrix_n,matrix_N,Z_perm)
                num+=1
                if p_value>alpha:
                    N110low=N110
                    break
                N110+=1
            tempa=max(matrix_n[0][0],N[0]-M101-M100-matrix_n[0][2],N[0]-M100-matrix_n[0][1]-matrix_n[0][2])
            tempb=min(N[0]-M101-M100,N[0]-M100-matrix_n[0][2],N[0]-matrix_n[0][1]-matrix_n[0][2])
            N110low2=tempa-matrix_n[0][0]
            N110up2=tempb
            #algorithm 4d
            min_N110=max(N110low2,N110low)
            if min_N110<=N110up2:
                tau=min_N110-M101+N210-M201+sum(pre_matrix_N[i][1]-matrix_N[i][2] for i in range(2,k))
                if tau<L:
                    L=tau

    return [L, U, num]




