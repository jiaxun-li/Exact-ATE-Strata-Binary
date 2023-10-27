
import Combined_Permute
import Hypergeo
import numpy as np
import matplotlib.pyplot as plt
import csv
from threading import Thread
from time import time
import Wald


matrix_N_list=[
    np.array([[8,15,0,7],[9,21,1,9],[7,26,1,6]]),
    np.array([[10,20,0,10],[7,1,25,7],[12,8,8,12]])
]
n_list=[
    [10,10,10],
    [20,30,10]
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

    ci_ws=[0,0]
    time_ws=0

    s=time()
    ci_permute_combine=Combined_Permute.confidence_interval_permute_combine(matrix_n,alpha,replication)
    e=time()
    time_permute_comb=round(e-s,5)

    s=time()
    ci_permute=[0,0]
    e=time()
    time_permute=0

    return [matrix_N,matrix_n,ci_wald,time_wald,ci_fast,time_fast,ci_ws,time_ws,ci_permute_combine,time_permute_comb,ci_permute,time_permute]

def simu_table(matrix_N_list: list,n_list: list, file_name: str,alpha:float=0.05,replication:int=100,simulation:int=100):
    L=len(matrix_N_list)
    for i in range(L):
        for j in range(simulation):
            res=simu_matrix_N(matrix_N_list[i],n_list[i],alpha,replication)
            with open(file_name,"a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(res)
simu_table(matrix_N_list,n_list,r"E:\CollegeLife\UCB first semester\Extended_Li_and_Ding\Paper_Code\csv\table2.csv")