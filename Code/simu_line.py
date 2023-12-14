#Simulation the Figures in Section 5.1

from Algo_for_extend_permute import Balanced
import Combined_Permute
import Hypergeo
import numpy as np
import matplotlib.pyplot as plt
import csv
from threading import Thread
from time import time
import os


def simulation_tau_2(N: list, n: list, tau1: int,tau2: int, file_name: str, alpha:float=0.05,replication: int=100,binom:float=0.5):

    potential_outcome1=np.zeros((N[0],2))
    if tau1>=0:
        for i in range(tau1):
            potential_outcome1[i]=[0,1]
    if tau1<=0:
        for i in range(abs(tau1)):
            potential_outcome1[i]=[1,0]
    temp=np.random.binomial(1,binom,N[0]-abs(tau1))
    for i in range(abs(tau1),N[0]):
        potential_outcome1[i][0]=temp[i-abs(tau1)]
    np.random.shuffle(temp)
    for i in range(abs(tau1),N[0]):
        potential_outcome1[i][1]=temp[i-abs(tau1)]
    np.random.shuffle(potential_outcome1)

    potential_outcome2=np.zeros((N[1],2))
    if tau2>=0:
        for i in range(tau2):
            potential_outcome2[i]=[0,1]
    if tau2<=0:
        for i in range(abs(tau2)):
            potential_outcome2[i]=[1,0]
    temp=np.random.binomial(1,binom,N[0]-abs(tau2))
    for i in range(abs(tau2),N[0]):
        potential_outcome2[i][0]=temp[i-abs(tau2)]
    np.random.shuffle(temp)
    for i in range(abs(tau2),N[0]):
        potential_outcome2[i][1]=temp[i-abs(tau2)]
    np.random.shuffle(potential_outcome2)
    pre_Z=[[]]*2
    for i in range(2):
        pre_Z[i]=np.block([np.ones(n[i]),np.zeros(N[i]-n[i])])
        np.random.shuffle(pre_Z[i])
    Z=[b for a in pre_Z for b in a]
    matrix_n=np.zeros((2,4),dtype=int)

    for i in range(sum(N)):
        if i<N[0]:
            if int(Z[i])==1 and potential_outcome1[i][1]==1:
                matrix_n[0][0]+=1
            elif int(Z[i])==1 and potential_outcome1[i][1]==0:
                matrix_n[0][1]+=1
            elif int(Z[i])==0 and potential_outcome1[i][0]==1:
                matrix_n[0][2]+=1
            else:
                matrix_n[0][3]+=1
        else:
            if int(Z[i])==1 and potential_outcome2[i-N[0]][1]==1:
                matrix_n[1][0]+=1
            elif int(Z[i])==1 and potential_outcome2[i-N[0]][1]==0:
                matrix_n[1][1]+=1
            elif int(Z[i])==0 and potential_outcome2[i-N[0]][0]==1:
                matrix_n[1][2]+=1
            else:
                matrix_n[1][3]+=1
    s=time()
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
    ci_permute=Balanced.confidence_interval_permute(matrix_n,alpha,replication)[:2]
    e=time()
    time_permute=round(e-s,5)

    with open(file_name,"a") as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow([matrix_n,ci_fast,ci_ws,ci_permute_combine,ci_permute,time_fast,time_ws,time_permute_comb,time_permute])
    return [ci_fast[1]-ci_fast[0],ci_ws[1]-ci_ws[0],ci_permute_combine[1]-ci_permute_combine[0],ci_permute[1]-ci_permute[0]]

def simulation_tau_3(N: list, n: list, tau1: int,tau2: int, tau3: int,file_name: str, alpha:float=0.05,replication: int=100,ws1:bool=True,binom=0.5):

    potential_outcome1=np.zeros((N[0],2))
    if tau1>=0:
        for i in range(tau1):
            potential_outcome1[i]=[0,1]
    if tau1<=0:
        for i in range(abs(tau1)):
            potential_outcome1[i]=[1,0]
    temp=np.random.binomial(1,binom,N[0]-abs(tau1))
    for i in range(abs(tau1),N[0]):
        potential_outcome1[i][0]=temp[i-abs(tau1)]
    np.random.shuffle(temp)
    for i in range(abs(tau1),N[0]):
        potential_outcome1[i][1]=temp[i-abs(tau1)]
    np.random.shuffle(potential_outcome1)

    potential_outcome2=np.zeros((N[1],2))
    if tau2>=0:
        for i in range(tau2):
            potential_outcome2[i]=[0,1]
    if tau2<=0:
        for i in range(abs(tau2)):
            potential_outcome2[i]=[1,0]
    temp=np.random.binomial(1,binom,N[1]-abs(tau2))
    for i in range(abs(tau2),N[1]):
        potential_outcome2[i][0]=temp[i-abs(tau2)]
    np.random.shuffle(temp)
    for i in range(abs(tau2),N[1]):
        potential_outcome2[i][1]=temp[i-abs(tau2)]
    np.random.shuffle(potential_outcome2)

    potential_outcome3=np.zeros((N[2],2))
    if tau3>=0:
        for i in range(tau3):
            potential_outcome3[i]=[0,1]
    if tau3<=0:
        for i in range(abs(tau3)):
            potential_outcome3[i]=[1,0]
    temp=np.random.binomial(1,binom,N[2]-abs(tau3))
    for i in range(abs(tau3),N[2]):
        potential_outcome3[i][0]=temp[i-abs(tau3)]
    np.random.shuffle(temp)
    for i in range(abs(tau3),N[2]):
        potential_outcome3[i][1]=temp[i-abs(tau3)]
    np.random.shuffle(potential_outcome3)


    pre_Z=[[]]*3
    for i in range(3):
        pre_Z[i]=np.block([np.ones(n[i]),np.zeros(N[i]-n[i])])
        np.random.shuffle(pre_Z[i])
    Z=[b for a in pre_Z for b in a]
    matrix_n=np.zeros((3,4),dtype=int)

    for i in range(sum(N)):
        if i<N[0]:
            if int(Z[i])==1 and potential_outcome1[i][1]==1:
                matrix_n[0][0]+=1
            elif int(Z[i])==1 and potential_outcome1[i][1]==0:
                matrix_n[0][1]+=1
            elif int(Z[i])==0 and potential_outcome1[i][0]==1:
                matrix_n[0][2]+=1
            else:
                matrix_n[0][3]+=1
        elif i<N[0]+N[1]:
            if int(Z[i])==1 and potential_outcome2[i-N[0]][1]==1:
                matrix_n[1][0]+=1
            elif int(Z[i])==1 and potential_outcome2[i-N[0]][1]==0:
                matrix_n[1][1]+=1
            elif int(Z[i])==0 and potential_outcome2[i-N[0]][0]==1:
                matrix_n[1][2]+=1
            else:
                matrix_n[1][3]+=1
        else:
            if int(Z[i])==1 and potential_outcome3[i-N[0]-N[1]][1]==1:
                matrix_n[2][0]+=1
            elif int(Z[i])==1 and potential_outcome3[i-N[0]-N[1]][1]==0:
                matrix_n[2][1]+=1
            elif int(Z[i])==0 and potential_outcome3[i-N[0]-N[1]][0]==1:
                matrix_n[2][2]+=1
            else:
                matrix_n[2][3]+=1
    s=time()
    ci_fast=Hypergeo.confidence_interval_hypergeo_fast(matrix_n,alpha)
    e=time()
    time_fast=round(e-s,5)
    if ws1:
        s=time()
        ci_ws=Hypergeo.confidence_interval_hypergeo_ws(matrix_n,alpha)
        e=time()
        time_ws=round(e-s,5)
    else:
        ci_ws=None
        time_ws=None


    s=time()
    ci_permute_combine=Combined_Permute.confidence_interval_permute_combine(matrix_n,alpha,replication)
    e=time()
    time_permute_comb=round(e-s,5)

    s=time()
    ci_permute=Balanced.confidence_interval_permute(matrix_n,alpha,replication)[:2]
    e=time()
    time_permute=round(e-s,5)

    with open(file_name,"a") as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow([matrix_n,ci_fast,ci_ws,ci_permute_combine,ci_permute,time_fast,time_ws,time_permute_comb,time_permute])
    if ci_ws!=None:
        return [ci_fast[1]-ci_fast[0],ci_ws[1]-ci_ws[0],ci_permute_combine[1]-ci_permute_combine[0],ci_permute[1]-ci_permute[0]]
    else:
        return [ci_fast[1]-ci_fast[0],None,ci_permute_combine[1]-ci_permute_combine[0],ci_permute[1]-ci_permute[0]]

def simulation_plot(tau_list: list, file_name: str, alpha:float=0.05,replication: int=100, simulation: int=100,ws1=True,binom=0.5):
    if len(tau_list)==2:
        for N in range(24,121,4):
            strata_N=[N//2,N//2]
            strata_n=[N//4,N//4]
            tau1=int(tau_list[0]*N/2)
            tau2=int(tau_list[1]*N/2)
            for i in range(simulation):
                simulation_tau_2(strata_N,strata_n,tau1,tau2,file_name,alpha,replication,binom)
    if len(tau_list)==3:
        for N in range(120,121,6):
            strata_N=[N//3,N//3,N//3]
            strata_n=[N//6,N//6,N//6]
            tau1=int(tau_list[0]*N/3)
            tau2=int(tau_list[1]*N/3)
            tau3=int(tau_list[2]*N/3)
            for i in range(simulation):
                simulation_tau_3(strata_N,strata_n,tau1,tau2,tau3,file_name,alpha,replication,ws1,binom)

# simulation_plot([0,0,0],r"\csv\tau000_test.csv",binom=0.5,ws1=False)