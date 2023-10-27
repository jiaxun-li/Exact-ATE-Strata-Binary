# Permutation and Combining Function

import numpy as np
from scipy.stats import chi2
import math
from itertools import product
import scipy.stats as stat
inf=10000

def fisher(list_p_values: list) -> float:
    l=len(list_p_values)
    for i in list_p_values:
        if i==0:
            return 0
    return 1-chi2.cdf(sum([-2*math.log(i) for i in list_p_values]),df=2*l)

def pearson(list_p_values: list) -> float:
    for i in range(len(list_p_values)):
        if list_p_values[i]==0:
            list_p_values[i]=0.0001
        if list_p_values[i]==1:
            list_p_values[i]=0.99999
    s, p=stat.combine_pvalues(list_p_values,'pearson')
    return p

def george(list_p_values: list) -> float:
    for i in range(len(list_p_values)):
        if list_p_values[i]==0:
            list_p_values[i]=0.0001
        if list_p_values[i]==1:
            list_p_values[i]=0.99999
    s, p=stat.combine_pvalues(list_p_values,'mudholkar_george')
    return p

def tippett(list_p_values: list) -> float:
    for i in range(len(list_p_values)):
        if list_p_values[i]==0:
            list_p_values[i]=0.0001
        if list_p_values[i]==1:
            list_p_values[i]=0.99999
    s, p=stat.combine_pvalues(list_p_values,'tippett')
    return p

def stouffer(list_p_values: list) -> float:
    for i in range(len(list_p_values)):
        if list_p_values[i]==0:
            list_p_values[i]=0.0001
        if list_p_values[i]==1:
            list_p_values[i]=0.99999
    s, p=stat.combine_pvalues(list_p_values,'stouffer')
    return p

def Z_generator(Ni: int, ni: int, replication: int, gen: callable=np.random) -> np.matrix:
    Z_perm=np.block([np.ones((replication,ni)),np.zeros((replication,Ni-ni))])
    for i in range(0,replication):
        gen.shuffle(Z_perm[i])
    return Z_perm


def filter_table(matrix_ni: list, matrix_Ni: list) -> bool:
    n11,n10,n01,n00=matrix_ni
    N = np.sum(matrix_Ni)   # total subjects
    return max(0,n11-matrix_Ni[1], matrix_Ni[0]-n01, matrix_Ni[0]+matrix_Ni[2]-n10-n01) <= min(matrix_Ni[0], n11, matrix_Ni[0]+matrix_Ni[2]-n01, N-matrix_Ni[1]-n01-n10)


def default_T(matrix_ni: list, matrix_Ni: list) -> float:
    Ni=np.sum(matrix_Ni)
    ni=matrix_ni[0]+matrix_ni[1]
    tau=(matrix_Ni[1]-matrix_Ni[2])/Ni-matrix_ni[0]/ni+matrix_ni[2]/(Ni-ni)
    return round(abs(tau),10)

def p_value_matrix(matrix_ni: list, matrix_Ni: list, Z_perm: np.array) -> float:
    n=matrix_ni[1]+matrix_ni[0] 
    N=np.sum(matrix_Ni)
    N11,N10,N01,N00=matrix_Ni
    replication=np.shape(Z_perm)[0]
    
    dat=np.zeros((N,2))
    dat[:N11,:]=1
    dat[N11:N11+N10,0]=1
    dat[N11:N11+N10,1]=0
    dat[N11+N10:N11+N10+N01,0]=0
    dat[N11+N10:N11+N10+N01,1]=1
    dat[N11+N10+N01:,:]=0
    
    tau_hat=np.dot(Z_perm,dat[:,0])/n-np.dot((1-Z_perm),dat[:,1])/(N-n)
    tau=(N10-N01)/N
    p=sum(np.round(abs(tau-tau_hat),10)>=default_T(matrix_ni,matrix_Ni))/replication
    return p

def Ni_generator(matrix_ni: list):
    n11,n10,n01,n00=matrix_ni
    N=sum(matrix_ni)
    res={}
    for i in range(min(N-n00, N-n10)+1):
        N11 = i
        for j in range(max(0, n01-N11), N-n00-N11+1):
            N01 = j
            for k in range(max(0, n11-N11), min(N-n10-N11, N-N11-N01)+1):
                N10 = k
                N00 = N-N11-N10-N01
                if filter_table(matrix_ni, [N11,N10,N01,N00]):
                    if N10-N01 not in res:
                        res[N10-N01]={}
                    if min(N10,N01) not in res[N10-N01]:
                        res[N10-N01][min(N10,N01)]=[]
                    res[N10-N01][min(N10,N01)].append([N11,N10,N01,N00])
    return res

def p_tau(matrix_ni,Z_perm):
    p_value={}
    Ni=np.sum(matrix_ni)
    ni=matrix_ni[0]+matrix_ni[1]
    matrix_Ni_dict=Ni_generator(matrix_ni)
    for tau in matrix_Ni_dict:
        p_value[tau]=0
        for min_N00_N11 in matrix_Ni_dict[tau]:
            for matrix_Ni in matrix_Ni_dict[tau][min_N00_N11]:
                p=p_value_matrix(matrix_ni,matrix_Ni,Z_perm)
                if p>=p_value[tau]:
                    p_value[tau]=p
    return p_value

def confidence_interval_permute_combine(matrix_n: list, alpha: float=0.05, replication: int=10000, combine: callable=fisher, T: callable=default_T) -> list:
    k=len(matrix_n)
    N=[sum(matrix_n[i]) for i in range(k)]
    n=[matrix_n[i][0]+matrix_n[i][1] for i in range(k)]
    L=np.sum(N)
    U=-np.sum(N)
    total_p_value=[{}]*k
    Z_perm=[Z_generator(N[i], n[i], replication) for i in range(k)]
    for i in range(k):
        total_p_value[i]=p_tau(matrix_n[i],Z_perm[i])
    for tau_list in product(*total_p_value):
        ATE=sum(tau_list)
        if ATE>U or ATE<L:
            p_value_list=[total_p_value[i][tau_list[i]] for i in range(k)]
            if combine(p_value_list)>=alpha:
                if ATE>U:
                    U=ATE
                if ATE<L:
                    L=ATE
    return [L,U]

# matrix_n=[
#     [8,21,3,25],
#     [8,14,2,24]
# ]
# print(confidence_interval_permute_combine(matrix_n,replication=10000))
