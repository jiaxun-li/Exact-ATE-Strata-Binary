# Extended Permutation method, with algorithm in Section A.3

import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
from itertools import product

def Z_generator(N: list, n: list, replication: int, gen: callable=np.random) -> np.matrix:
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
    shuffle=gen.shuffle
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
        a dictionary of dictionary of potential outcome tables algebraically consistent with the observed table. Key values are assigned using Theorem A.6 and A.7
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
                        res[N10-N01]={}
                    if min(N10,N01) not in res[N10-N01]:
                        res[N10-N01][min(N10,N01)]=[]
                    res[N10-N01][min(N10,N01)].append([N11,N10,N01,N00])
    return res

def Ni_core_generator(matrix_Ni_dict: dict, Ni: int):
    ''' 
    Generate the potential outcome tables in Step2(b) in Algorithm A.4.

    Parameters: 
    ---------- 
    matrix_Ni_dict : dict
        the dictionary of potential outcome tables, generated from the  function Ni_generator

    Ni : int
        The  size of subjects
    Returns:
    --------
    matrix_Ni_dict_core : dict,
        a dictionary of potential outcome tables in Step 2(b). Key value is the average treatment effect.
    '''
    matrix_Ni_dict_core={}
    for tau in matrix_Ni_dict:
        temp_N11_N00=[]
        min_jiaxun1=10000
        min_jiaxun2=10000
        for min_N10_N01 in range(Ni//2):
            if min_N10_N01 not in matrix_Ni_dict[tau]:
                continue
            for matrix_Ni in matrix_Ni_dict[tau][min_N10_N01]:
                N11,N10,N01,N00=matrix_Ni
                if (N11+N10)%2==1 and abs(N11-N00)<min_jiaxun1:
                    min_jiaxun1=abs(N11-N00)
                if (N11+N10)%2==0 and abs(N11-N00)<min_jiaxun2:
                    min_jiaxun2=abs(N11-N00)
            for matrix_Ni in matrix_Ni_dict[tau][min_N10_N01]:
                N11,N10,N01,N00=matrix_Ni
                if (N11+N10)%2==1 and abs(N11-N00)>min_jiaxun1:
                    continue
                if (N11+N10)%2==0 and abs(N11-N00)>min_jiaxun2:
                    continue
                if abs(N11-N00) in temp_N11_N00:
                    continue
                temp_N11_N00.append(abs(N11-N00))
                if tau not in matrix_Ni_dict_core:
                    matrix_Ni_dict_core[tau]=[]
                matrix_Ni_dict_core[tau].append(matrix_Ni)
    return matrix_Ni_dict_core

def Ni_jiaxun_generator(matrix_Ni_dict: dict, Ni: int, tau: int):
    '''
    Generate the potential outcome tables algebraically consistent with the the observed table and has the lowest value of |N11-N00| with the given tau(see Theorem A.7).

    Parameters:
    ----------
    matrix_Ni_dict : dict
        the dictionary of potential outcome tables, generated from the  function Ni_generator
    Ni : int
        The  size of subjects
    tau : int
        Ni times the average treatment effect

    Returns:
    --------
    matrix_Ni_list_jiaxun : list,
        a lists of potential outcome tables algebraically consistent with the the observed table and has the lowest value of |N11-N00| with the given tau (see Theorem A.7).
    '''
    matrix_Ni_list_jiaxun=[]
    min_jiaxun1=10000
    min_jiaxun2=10000
    for min_N10_N01 in range(Ni//2):
        if min_N10_N01 not in matrix_Ni_dict[tau]:
            continue
        for matrix_Ni in matrix_Ni_dict[tau][min_N10_N01]:
            N11,N10,N01,N00=matrix_Ni
            if (N11+N10)%2==1 and abs(N11-N00)<min_jiaxun1:
                min_jiaxun1=abs(N11-N00)
            if (N11+N10)%2==0 and abs(N11-N00)<min_jiaxun2:
                min_jiaxun2=abs(N11-N00)
        for matrix_Ni in matrix_Ni_dict[tau][min_N10_N01]:
            N11,N10,N01,N00=matrix_Ni
            if (N11+N10)%2==1 and abs(N11-N00)>min_jiaxun1:
                continue
            if (N11+N10)%2==0 and abs(N11-N00)>min_jiaxun2:
                continue
            matrix_Ni_list_jiaxun.append(matrix_Ni)
    return matrix_Ni_list_jiaxun


def confidence_interval_permute(matrix_n,alpha=0.05,replication=10000):
    ''' 
    Generate the confidence interval for a observed table with Algorithm A.4

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
    [L, U, num] : list
        confidence bound and the number of permutation tests. The first element is the lower bound, the second element is the upper bound. The last number is the number of permutation tests.
    '''
    k=len(matrix_n)
    N=[np.sum(matrix_n[i]) for i in range(k)]
    n=[matrix_n[i][0]+matrix_n[i][1] for i in range(k)]

    tau_temp=[]

    L=np.sum(matrix_n)
    U=-np.sum(matrix_n)
    Z_perm=Z_generator(N, n, replication)

    matrix_N_core_temp=[]
    matrix_N_core_dict={}
    matrix_N_temp={}
    for i in range(k):
        matrix_N_temp[i]=Ni_generator(matrix_n[i])

        tau_temp.append(range(min(matrix_N_temp[i].keys()),max(matrix_N_temp[i].keys())+1))

        matrix_N_core_temp.append(Ni_core_generator(matrix_N_temp[i],N[i]))

    tau_vector_list=list(product(*tau_temp))
    for tau_vector in tau_vector_list:
        temp=[]
        for i in range(k):
            temp.append(matrix_N_core_temp[i][tau_vector[i]])
        matrix_N_core_dict[tau_vector]=list(product(*temp))
    num=0
    for tau_vector in tau_vector_list:
        tau=sum(tau_vector)
        if tau<=U and tau>=L:
                continue
        for matrix_N in matrix_N_core_dict[tau_vector]:
            p=p_value_matrix(matrix_n,matrix_N,Z_perm)
            num+=1
            if p>alpha:
                if tau>U:
                    U=tau
                if tau<L:
                    L=tau
                break
    if U%2==1:
        tau_interest=U+1
        ac=0
        for tau_vector in tau_vector_list:
            if ac==1:
                break
            if 0 not in tau_vector:
                continue
            if sum(tau_vector)!= tau_interest:
                continue
            i=tau_vector.index(0)
            matrix_N_jiaxun_temp=[]
            for j in range(k):
                if j!=i:
                    matrix_N_jiaxun_temp.append(Ni_jiaxun_generator(matrix_N_temp[j],N[j],tau_vector[j]))
                else:
                    if 0 in matrix_N_temp[i][0]:
                        matrix_N_jiaxun_temp.append(matrix_N_temp[i][0][0])
                    else:
                        matrix_N_jiaxun_temp.append([])
            if [] not in matrix_N_jiaxun_temp:
                for matrix_N in product(*matrix_N_jiaxun_temp):
                    p=p_value_matrix(matrix_n,matrix_N,Z_perm)
                    num+=1
                    if p>alpha:
                        U=tau_interest
                        ac=1
                        break
            if ac==1:
                break
            matrix_N_jiaxun_temp=[]
            for j in range(k):
                if tau_vector[j]!=0:
                    matrix_N_jiaxun_temp.append(Ni_jiaxun_generator(matrix_N_temp[j],N[j],tau_vector[j]))
                else:
                    if 1 in matrix_N_temp[j][0]:
                        matrix_N_jiaxun_temp.append(matrix_N_temp[j][0][1])
                    else:
                        matrix_N_jiaxun_temp.append([])
            if [] not in matrix_N_jiaxun_temp:
                for matrix_N in product(*matrix_N_jiaxun_temp):
                    p=p_value_matrix(matrix_n,matrix_N,Z_perm)
                    num+=1
                    if p>alpha:
                        U=tau_interest
                        ac=1
                        break

    if L%2==1:
        tau_interest=L-1
        ac=0
        for tau_vector in tau_vector_list:
            if ac==1:
                break
            if 0 not in tau_vector:
                continue
            if sum(tau_vector)!= tau_interest:
                continue
            i=tau_vector.index(0)
            matrix_N_jiaxun_temp=[]
            for j in range(k):
                if j!=i:
                    matrix_N_jiaxun_temp.append(Ni_jiaxun_generator(matrix_N_temp[j],N[j],tau_vector[j]))
                else:
                    if 0 in matrix_N_temp[i][0]:
                        matrix_N_jiaxun_temp.append(matrix_N_temp[i][0][0])
                    else:
                        matrix_N_jiaxun_temp.append([])
            if [] not in matrix_N_jiaxun_temp:
                for matrix_N in product(*matrix_N_jiaxun_temp):
                    tau=sum(matrix_N[i][1]-matrix_N[i][2] for i in range(k))
                    num+=1
                    if tau!=tau_interest:
                        continue
                    p=p_value_matrix(matrix_n,matrix_N,Z_perm)
                    if p>alpha:
                        L=tau_interest
                        ac=1
                        break
            if ac==1:
                break

            matrix_N_jiaxun_temp=[]
            for j in range(k):
                if tau_vector[j]!=0:
                    matrix_N_jiaxun_temp.append(Ni_jiaxun_generator(matrix_N_temp[j],N[j],tau_vector[j]))
                else:
                    if 1 in matrix_N_temp[j][0]:
                        matrix_N_jiaxun_temp.append(matrix_N_temp[j][0][1])
                    else:
                        matrix_N_jiaxun_temp.append([])
            if [] not in matrix_N_jiaxun_temp:
                for matrix_N in product(*matrix_N_jiaxun_temp):
                    num+=1
                    p=p_value_matrix(matrix_n,matrix_N,Z_perm)
                    if p>alpha:
                        L=tau_interest
                        ac=1
                        break
    return [L, U,num]

def binary_search(lb, ub, f):
    """
    binary search in [lb, ub)
    """
    s = 0
    l = lb
    r = ub
    while l < r - 1:
        c = math.floor((l + r) / 2)
        f_c = f(c)
        if f_c[0] == 1:
            l = c
        else:
            r = c
        s += f_c[1]
    return [l, s]

def binary_search_opp(lb, ub, f):
    """
    binary search in (lb, ub]
    """
    s = 0
    l = lb
    r = ub
    while r > l + 1:
        c = math.ceil((l + r) / 2)
        f_c = f(c)
        if f_c[0] == 1:
            r = c
        else:
            l = c
        s += f_c[1]
    return [r, s]

def confidence_interval_permute_one_strata(matrix_n,alpha=0.05,replication=100000):
    ''' 
    Generate the confidence interval for a observed table with one strata with Algorithm A.4. 

    Parameters: 
    ---------- 
    matrix_n : list of four ints
        the observed table, in the order
        [ni_11, ni_10, ni_01, ni_00]
    alpha : float
        1-confidence level, default is 0.05
    replication : int
        number of replications for the Monte Carlo permutation test
    Returns:
    --------
    [L, U, num] : list
        confidence bound and the number of permutation tests. The first element is the lower bound, the second element is the upper bound. The last number is the number of permutation tests.
    '''
    N=np.sum(matrix_n)
    n=matrix_n[0]+matrix_n[1]

    Z_perm=Z_generator([N], [n], replication)

    matrix_N_temp=Ni_generator(matrix_n)
    tau_list=range(min(matrix_N_temp.keys()),max(matrix_N_temp.keys())+1)
    matrix_N_core_dict=Ni_core_generator(matrix_N_temp,N)
    num=0
    def test_tau(tau):
        num=0
        if tau not in matrix_N_core_dict:
            return [0,0]
        for matrix_N in matrix_N_core_dict[tau]:
            p=p_value_matrix([matrix_n],[matrix_N],Z_perm)
            num+=1
            if p>=alpha:
                return [1,num]
        return [0,num]
    tauhat=2*(matrix_n[0]-matrix_n[2])

    k2=tauhat
    k1=matrix_n[3]+matrix_n[0]-N-1
    L,num1=binary_search_opp(k1,k2,test_tau)

    k1=tauhat
    k2=matrix_n[3]+matrix_n[0]+1
    U,num2=binary_search(k1,k2,test_tau)
    return [L,U,num1+num2]

# matrix_n=[[5, 5, 5, 5],
#           [5,5,5,5]]
# print(confidence_interval_permute(matrix_n))