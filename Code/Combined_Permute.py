# Permutation and Combining Function

import numpy as np
from scipy.stats import chi2
import math
from itertools import product
import scipy.stats as stat
inf=10000

def fisher(list_p_values: list) -> float:
    '''
    Fisher's combining function for the log of independent P-values
    
    Parameter
    ---------
    list_p_values : list
        vector of independent P-values

    Returns
    -------
    P : float
        combined P-value
    '''
    l=len(list_p_values)
    for i in list_p_values:
        if i==0:
            return 0
    return 1-chi2.cdf(sum([-2*math.log(i) for i in list_p_values]),df=2*l)

def pearson(list_p_values: list) -> float:
    '''
    Pearson's combining function for the log of independent P-values
    
    Parameter
    ---------
    list_p_values : list
        vector of independent P-values

    Returns
    -------
    P : float
        combined P-value
    '''
    for i in range(len(list_p_values)):
        if list_p_values[i]==0:
            list_p_values[i]=0.0001
        if list_p_values[i]==1:
            list_p_values[i]=0.99999
    s, p=stat.combine_pvalues(list_p_values,'pearson')
    return p

def george(list_p_values: list) -> float:
    '''
    Mudholkar's and  George's combining function for the log of independent P-values
    
    Parameter
    ---------
    list_p_values : list
        vector of independent P-values

    Returns
    -------
    P : float
        combined P-value
    '''
    for i in range(len(list_p_values)):
        if list_p_values[i]==0:
            list_p_values[i]=0.0001
        if list_p_values[i]==1:
            list_p_values[i]=0.99999
    s, p=stat.combine_pvalues(list_p_values,'mudholkar_george')
    return p

def tippett(list_p_values: list) -> float:
    '''
    Tippett's combining function for the log of independent P-values
    
    Parameter
    ---------
    list_p_values : list
        vector of independent P-values

    Returns
    -------
    P : float
        combined P-value
    '''
    for i in range(len(list_p_values)):
        if list_p_values[i]==0:
            list_p_values[i]=0.0001
        if list_p_values[i]==1:
            list_p_values[i]=0.99999
    s, p=stat.combine_pvalues(list_p_values,'tippett')
    return p

def stouffer(list_p_values: list) -> float:
    '''
    Stoffer's combining function for the log of independent P-values
    
    Parameter
    ---------
    list_p_values : list
        vector of independent P-values

    Returns
    -------
    P : float
        combined P-value
    '''
    for i in range(len(list_p_values)):
        if list_p_values[i]==0:
            list_p_values[i]=0.0001
        if list_p_values[i]==1:
            list_p_values[i]=0.99999
    s, p=stat.combine_pvalues(list_p_values,'stouffer')
    return p

def Z_generator(Ni: int, ni: int, replication: int) -> np.matrix:
    '''
    Randomly generate the treatment assignment 'replication' times
    
    Parameter
    ---------
    Ni : int
        the size of the subjects
    ni : int
        the size of the treatment group
    replication : int
        number of generation times

    Returns
    -------
    Z_perm : np.matrix, Ni*replication size matrix
        The generated treatment assignment, each row is a random generation. 1 represents treatment and 0 represents control
    '''
    Z_perm=np.block([np.ones((replication,ni)),np.zeros((replication,Ni-ni))])
    for i in range(0,replication):
        np.random.shuffle(Z_perm[i])
    return Z_perm


def filter_table(matrix_ni: list, matrix_Ni: list) -> bool:
    ''' 
    Check whether summary table matrix_Ni of binary outcomes is consistent with observed table matrix_ni

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
    n11,n10,n01,n00=matrix_ni
    N = np.sum(matrix_Ni)   # total subjects
    return max(0,n11-matrix_Ni[1], matrix_Ni[0]-n01, matrix_Ni[0]+matrix_Ni[2]-n10-n01) <= min(matrix_Ni[0], n11, matrix_Ni[0]+matrix_Ni[2]-n01, N-matrix_Ni[1]-n01-n10)


def default_T(matrix_ni: list, matrix_Ni: list) -> float:
    ''' 
    The default test statistic for permutation test:

    |n11/n-n01/(N-n)-(N10-N01)/N|

    Parameters: 
    ---------- 
    matrix_ni : list of four ints
        the observed table in one strata, 
        in the order n_11, n_10, n_01, n_00
    matrix_Ni : list of four ints 
        the potential outcome table in one strata
        in the order N_11, N_10, N_01, N_00 

    Returns:
    --------
    stat: float,
        The test statistic
    '''
    Ni=np.sum(matrix_Ni)
    ni=matrix_ni[0]+matrix_ni[1]
    tau=(matrix_Ni[1]-matrix_Ni[2])/Ni-matrix_ni[0]/ni+matrix_ni[2]/(Ni-ni)
    return round(abs(tau),10)

def p_value_matrix(matrix_ni: list, matrix_Ni: list, Z_perm: np.array) -> float:
    ''' 
    The permutation p-value that the observed table matrix_ni comes from the potential outcome table matrix_Ni.

    Parameters: 
    ---------- 
    matrix_ni : list of four ints
        the observed table in one strata, 
        in the order n_11, n_10, n_01, n_00
    matrix_Ni : list of four ints 
        the table of counts of subjects in one strata with each combination of potential outcomes, 
        in the order N_11, N_10, N_01, N_00 
    Z_perm: array, Ni*replication size matrix
        The generated treatment assignment, each row is a random generation. 1 represents treatment and 0 represents control
    Returns:
    --------
    P : float,
        The p-value for the potential outcome table matrix_Ni
    '''
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
    ''' 
    Generate the potential outcome tables algebraically consistent the the observed table

    Parameters: 
    ---------- 
    matrix_ni : list of four ints
        the observed table in one strata, 
        in the order n_11, n_10, n_01, n_00
    Returns:
    --------
    res : dict,
        a dictionary of potential outcome tables algebraically consistent with the observed table, key-values are the average treatment effect.
    '''

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
                        res[N10-N01]=[]
                    res[N10-N01].append([N11,N10,N01,N00])
    return res

def p_tau(matrix_ni,Z_perm):

    ''' 
    Generate the permutation p-value that the observed table matrix_ni has a average treatment effect tau, for each possible values of tau

    Parameters: 
    ---------- 
    matrix_ni : list of four ints
        the observed table in one strata, 
        in the order n_11, n_10, n_01, n_00
    Z_perm: array, Ni*replication size matrix
        The generated treatment assignment, each row is a random generation. 1 represents treatment and 0 represents control
    Returns:
    --------
    p_value : dict of floats
        The p-value for each possible value of the average treatment effect.
    '''

    p_value={}
    Ni=np.sum(matrix_ni)
    ni=matrix_ni[0]+matrix_ni[1]
    matrix_Ni_dict=Ni_generator(matrix_ni)
    for tau in matrix_Ni_dict:
        p_value[tau]=0
        for matrix_Ni in matrix_Ni_dict[tau]:
            p=p_value_matrix(matrix_ni,matrix_Ni,Z_perm)
            if p>=p_value[tau]:
                p_value[tau]=p
    return p_value

def confidence_interval_permute_combine(matrix_n: list, alpha: float=0.05, replication: int=10000, combine: callable=fisher) -> list:

    ''' 
    Generate the confidence interval for a observed table with the method in Section 4.4.

    Parameters: 
    ---------- 
    matrix_n : a list of lists, each individual list has four ints
        the stratified observed table, each strata in the order
        [ni_11, ni_10, ni_01, ni_00]
    alpha : float
        1-confidence level, default is 0.05
    replication : int
        number of replications for the Monte Carlo permutation test
    combine: callable
        the p-value combining function
    Returns:
    --------
    [L, U] : list
        confidence bound, the first element is the lower bound, the second element is the upper bound
    '''

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
