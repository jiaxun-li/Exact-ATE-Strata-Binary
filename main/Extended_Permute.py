# Extend Permutation method

import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
from itertools import product

def Z_generator(N: list, n: list, replication: int) -> list:
    '''
    Randomly generate the treatment assignment 'replication' times
    
    Parameter
    ---------
    N : list of ints
        a list of stratum-wise subject sizes
    n : list of int
        a list of stratum-wise treatment group sizes
    replication : int
        number of generation times

    Returns
    -------
    Z_perm : list of np.matrix, size of each matrix is N[i]*replication 
        A list of the generated treatment assignment. Each element represents random generations of the treatment assignment in one strata with each row being a random generation. 1 represents treatment and 0 represents control.
    '''
    k=len(N)
    Z_perm=[[]]*k
    for i in range(k):
        Z_perm[i]=np.block([np.ones((replication,n[i])),np.zeros((replication,N[i]-n[i]))])
    shuffle=np.random.shuffle
    for i in range(k):
        for j in range(replication):
            shuffle(Z_perm[i][j])
    return Z_perm


def filter_table(matrix_ni: list, matrix_Ni: list) -> bool:
    ''' 
    Check whether summary table matrix_Ni of binary outcomes is consistent with observed table matrix_ni (one-strata case)

    implements the test in Theorem 1 of Li and Ding (2016):
    \max \{0,n_{11}-N_{01}, N_{11}-n_{01}, N_{10}+N_{11}-n_{10}-n_{01} \} 
    \le 
    \min \{N_{11}, n_{11}, N_{10}+N_{11}-n_{01}, N-N_{01}-n_{01}-n_{10} \} 

    Parameters: 
    ---------- 
    matrix_ni : list of four ints
        the observed table in one strata, 
        in the order n_11, n_10, n_01, n_00
    matrix_Ni : list of four ints 
        the table of counts of subjects in one strata with each combination of potential outcomes, 
        in the order N_11, N_10, N_01, N_00 

    Returns:
    --------
    ok : bool True if table is consistent with the data 
    '''
    return max(0,matrix_ni[0]-matrix_Ni[1], matrix_Ni[0]-matrix_ni[2], matrix_Ni[0]+matrix_Ni[2]-matrix_ni[1]-matrix_ni[2]) <= min(matrix_Ni[0], matrix_ni[0], matrix_Ni[0]+matrix_Ni[2]-matrix_ni[2], np.sum(matrix_Ni)-matrix_Ni[1]-matrix_ni[1]-matrix_ni[2])



def default_T(matrix_n: list, matrix_N: list) -> float:
    ''' 
    The default test statistic for permutation test:

    |\sum_{i=1}^k[(Ni/N)*(ni_11/ni-ni_01/(Ni-ni)-(Ni_10-Ni_01)/Ni)]|

    Parameters:
    ----------
    matrix_n : a list of lists, each individual list has four ints
        the stratified observed table, each strata in the order
        [ni_11, ni_10, ni_01, ni_00]
    matrix_N : a list of lists, each individual list has four ints
        the stratified potential outcome table, each strata in the order
        [Ni_11, Ni_10, Ni_01, Ni_00]

    Returns:
    --------
    stat: float,
        The test statistic
    '''
    k=len(matrix_N)
    N=[np.sum(matrix_N[i]) for i in range(k)]
    n=[matrix_n[i][0]+matrix_n[i][1] for i in range(k)]
    tau_hat=sum(N[i]/np.sum(N)*((matrix_N[i][1]-matrix_N[i][2])/N[i]-matrix_n[i][0]/n[i]+matrix_n[i][2]/(N[i]-n[i])) for i in range(k))
    return round(abs(tau_hat),10)

def p_value_matrix(matrix_n: list, matrix_N: list, Z_perm: list) -> float:
    ''' 
    The permutation p-value that the observed table matrix_n comes from the potential outcome table matrix_N.

    Parameters: 
    ---------- 
    matrix_n : a list of lists, each individual list has four ints
        the stratified observed table, each strata in the order
        [ni_11, ni_10, ni_01, ni_00]
    matrix_N : a list of lists, each individual list has four ints
        the stratified potential outcome table, each strata in the order
        [Ni_11, Ni_10, Ni_01, Ni_00]
    Z_perm: array, Ni*replication size matrix
        The generated treatment assignment, each row is a random generation. 1 represents treatment and 0 represents control
    Returns:
    --------
    P : float,
        The p-value for the potential outcome table matrix_Ni
    '''
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
    ''' 
    Generate the potential outcome tables algebraically consistent the the observed table

    Parameters: 
    ---------- 
    matrix_ni : list of four ints
        the observed table in one strata, 
        in the order n_11, n_10, n_01, n_00
    Returns:
    --------
    res : list,
        a list of potential outcome tables algebraically consistent with the observed table.
    '''
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

def confidence_interval_permute(matrix_n,alpha=0.05,replication=10000):
    ''' 
    Generate the confidence interval for a observed table with the method in Section 4.3

    Parameters: 
    ---------- 
    matrix_n : a list of lists, each individual list has four ints
        the stratified observed table, each strata in the order
        [ni_11, ni_10, ni_01, ni_00]
    alpha : float
        1-confidence level, default is 0.05
    replication : int
        number of replications for the Monte Carlo permutation test
    Returns:
    --------
    [L, U] : list
        confidence bound, the first element is the lower bound, the second element is the upper bound
    '''
    k=len(matrix_n)
    N=[np.sum(matrix_n[i]) for i in range(k)]
    n=[matrix_n[i][0]+matrix_n[i][1] for i in range(k)]
    L=np.sum(matrix_n)
    U=-np.sum(matrix_n)
    Z_perm=Z_generator(N, n, replication)
    matrix_N_list=[]
    s=1
    for i in range(k):
        temp=Ni_generator(matrix_n[i])
        matrix_N_list.append(temp)
        s*=len(temp)
    j=1
    temp=0
    for matrix_N in product(*matrix_N_list):
        tau=sum(matrix_N[i][1]-matrix_N[i][2] for i in range(k))
        j+=1
        temp=j/s
        if L<=tau and tau<=U:
            continue
        p=p_value_matrix(matrix_n,matrix_N,Z_perm)
        if p<=alpha:
            continue
        if tau<L:
            L=tau
        if tau>U:
            U=tau

    return [L, U]


# matrix_n=[
#     [10,10,1,10],
#     [10,10,10,10]
# ]
# print(confidence_interval_permute(matrix_n))