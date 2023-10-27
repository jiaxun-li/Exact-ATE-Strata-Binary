#Simulate the table in Section 5.1

import Extended_Permute
import Combined_Permute
import Hypergeo
import numpy as np
import matplotlib.pyplot as plt
import csv
from threading import Thread
from time import time
import Wald


matrix_N_list=[
    np.array([[10,10,10,10],[10,10,10,10]]),
    np.array([[3,8,4,5],[0,19,1,0]]),
    np.array([[3,23,2,2],[4,2,30,4]]),
    np.array([[2,24,0,4],[1,26,2,1]]),
    np.array([[1,0,9,0],[0,40,0,0]]),
    np.array([[5,5,5,5],[20,50,2,8]]),
    np.array([[2,12,0,1],[2,55,1,2]]),
    np.array([[2,2,12,4],[3,64,1,2]]),
    np.array([[0,16,0,4],[3,9,1,7],[5,5,5,5]]),
    np.array([[1,13,1,0],[0,18,0,2],[0,20,0,5]]),
    np.array([[0,19,1,0],[3,4,4,4],[0,2,18,0]]),
    np.array([[3,2,2,3],[4,0,10,6],[5,23,3,9]])
]
n_list=[
    [10,10],
    [15,15],
    [5,30],
    [5,25],
    [5,20],
    [15,60],
    [10,40],
    [5,60],
    [5,10,15],
    [10,10,10],
    [5,5,5],
    [5,5,30]
]



def simu_matrix_N(matrix_N: list, n: list, alpha:float=0.05,replication:int=100):
    k=len(matrix_N)
    matrix_n=np.zeros((k,4),dtype=int)
    N=np.sum(matrix_N)
    for i in range(k):
        N11,N10,N01,N00=matrix_N[i]
        Ni=sum(matrix_N[i])
        
        N11_t=N10_t=N01_t=N00_t=0
        experiment=[0]*Ni
        experiment[:n[i]]=[1]*n[i]
        np.random.shuffle(experiment)
        
        N11_t=sum(experiment[:N11])
        N10_t=sum(experiment[N11:N11+N10])
        N01_t=sum(experiment[N11+N10:N11+N10+N01])
        N00_t=sum(experiment[N11+N10+N01:])

        matrix_n[i][0]=N10_t+N11_t
        matrix_n[i][1]=N00_t+N01_t
        matrix_n[i][2]=N01-N01_t+N11-N11_t
        matrix_n[i][3]=N00-N00_t+N10-N10_t
    s=time()
    ci_wald=Wald.wald_interval(matrix_n,alpha)
    ci_wald[0]*=N
    ci_wald[1]*=N
    e=time()
    time_wald=round(e-s,5)

    ci_fast=Hypergeo.confidence_interval_hypergeo_fast(matrix_n,alpha)
    e=time()
    time_fast=round(e-s,5)

    s=time()
    ci_ws=Hypergeo.confidence_interval_hypergeo_ws(matrix_n,alpha)
    e=time()
    time_ws=round(e-s,5)

    s=time()
    ci_permute_combine=Combined_Permute.confidence_interval_permute_combine(matrix_n,alpha,replication)
    e=time()
    time_permute_comb=round(e-s,5)

    s=time()
    ci_permute=Extended_Permute.confidence_interval_permute(matrix_n,alpha,replication)
    e=time()
    time_permute=round(e-s,5)

    return [matrix_N,matrix_n,ci_wald,time_wald,ci_fast,time_fast,ci_ws,time_ws,ci_permute_combine,time_permute_comb,ci_permute,time_permute]

def simu_table(matrix_N_list: list,n_list: list, file_name: str,alpha:float=0.05,replication:int=100,simulation:int=1):
    L=len(matrix_N_list)
    for i in range(L):
        for j in range(simulation):
            res=simu_matrix_N(matrix_N_list[i],n_list[i],alpha,replication)
            with open(file_name,"a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(res)
simu_table(matrix_N_list,n_list,r"E:\CollegeLife\UCB first semester\Extended_Li_and_Ding\Paper_Code\csv\table4.csv")