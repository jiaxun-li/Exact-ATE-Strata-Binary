# Extended Permutation method, with algorithm in Section A.2

import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
from itertools import product
import copy

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

def bound_difference_N10_N01(matrix_ni: list, N11: int, N00: int, is_upper: bool):
    N=np.sum(matrix_ni)
    n11,n10,n01,n00=matrix_ni
    if is_upper==False:
        return max(2*n11-N-N11+N00,-N+N11+N00,N-N11-N00-2*n01-2*n10,N11-N-N00+2*n00)
    if is_upper==True:
        return min(N11+N-N00-2*n01,N+N11+N00-2*n01-2*n10,N-N11-N00,N-N11+N00-2*n10)

def Ni_generator(matrix_ni: list) -> list:
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

def N_generator_partial_balanced(matrix_n: list, l: int) -> tuple:
    N=[np.sum(matrix_n[i]) for i in range(l)]
    temp=[]
    for i in range(l):
        n11,n10,n01,n00=matrix_n[i]
        res=[]
        for N11 in range(n01+n11+1):
            for N00 in range(N[i]-n01-n11+1):
                res.append([N11,N00])
        temp.append(res)
    matrix_N_list=[]
    for half_matrix_N in product(*temp):
        matrix_N=[]

        lower=[bound_difference_N10_N01(matrix_n[i],half_matrix_N[i][0], half_matrix_N[i][1], is_upper=False) for i in range(l)]

        upper=[bound_difference_N10_N01(matrix_n[i],half_matrix_N[i][0], half_matrix_N[i][1], is_upper=True) for i in range(l)]
        for i in range(l):
            N11=half_matrix_N[i][0]
            N00=half_matrix_N[i][1]
            N10=(lower[i]+N[i]-N11-N00)//2
            N01=N[i]-N11-N00-N10
            matrix_N.append([N11,N10,N01,N00])
        strata_index=0
        matrix_N_list.append(copy.deepcopy(matrix_N))
        while True:
            if matrix_N[strata_index][1]-matrix_N[strata_index][2]>=upper[strata_index]:
                strata_index+=1
            else:
                matrix_N[strata_index][1]+=1
                matrix_N[strata_index][2]-=1

                matrix_N_list.append(copy.deepcopy(matrix_N))
            if strata_index==l:
                break
    return matrix_N_list


def confidence_interval_permute(matrix_n,alpha=0.05,replication=10000):
    k=len(matrix_n)
    l=0
    for i in range(k):
        if matrix_n[i][0]+matrix_n[i][1]==matrix_n[i][2]+matrix_n[i][3]:
            l+=1
        else:
            break
    N=[np.sum(matrix_n[i]) for i in range(k)]
    n=[matrix_n[i][0]+matrix_n[i][1] for i in range(k)]
    L=np.sum(matrix_n)
    U=-np.sum(matrix_n)
    Z_perm=Z_generator(N, n, replication)
    num=0
    temp=[]
    temp.append(N_generator_partial_balanced(matrix_n,l))
    for i in range(l,k):
        temp.append(Ni_generator(matrix_n[i]))
    for pre_matrix_N in product(*temp):

        matrix_N=[pre_matrix_N[0][i] for i in range(l)]
        for i in range(l,k):
            matrix_N.append(pre_matrix_N[i])
        tau=sum(matrix_N[i][1]-matrix_N[i][2] for i in range(k))
        num+=1
        if L<=tau and tau<=U:
            continue
        p=p_value_matrix(matrix_n,matrix_N,Z_perm)
        if p<=alpha:
            continue
        if tau<L:
            L=tau
        if tau>U:
            U=tau

    return [L,U,num]


# matrix_n=[[5, 5, 5, 5],
#           [5,5,5,5]]
# print(confidence_interval_permute(matrix_n))