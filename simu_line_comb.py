import Combined_Permute
import numpy as np
import matplotlib.pyplot as plt
import csv
from threading import Thread
from time import time


def simulation_tau_2(N: list, n: list, tau1: int,tau2: int, alpha:float=0.05,replication: int=100):

    potential_outcome1=np.zeros((N[0],2))
    if tau1>=0:
        for i in range(tau1):
            potential_outcome1[i]=[0,1]
    if tau1<=0:
        for i in range(abs(tau1)):
            potential_outcome1[i]=[1,0]
    temp=np.random.binomial(1,0.5,N[0]-abs(tau1))
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
    temp=np.random.binomial(1,0.5,N[0]-abs(tau2))
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
    
    total_p_value=Combined_Permute.p_permute(matrix_n,replication)

    ci_fisher=Combined_Permute.ci_combine(total_p_value,Combined_Permute.fisher,alpha)
    width_fisher=ci_fisher[1]-ci_fisher[0]

    ci_pearson=Combined_Permute.ci_combine(total_p_value,Combined_Permute.pearson,alpha)
    width_pearson=ci_pearson[1]-ci_pearson[0]

    ci_george=Combined_Permute.ci_combine(total_p_value,Combined_Permute.george,alpha)
    width_george=ci_george[1]-ci_george[0]

    ci_tippet=Combined_Permute.ci_combine(total_p_value,Combined_Permute.tippett,alpha)
    width_tippet=ci_tippet[1]-ci_tippet[0]

    ci_stouffer=Combined_Permute.ci_combine(total_p_value,Combined_Permute.stouffer,alpha)
    width_stouffer=ci_stouffer[1]-ci_stouffer[0]

    ci_wright=Combined_Permute.ci_wright(total_p_value,Combined_Permute.stouffer,alpha)
    width_wright=ci_wright[1]-ci_wright[0]


    return [matrix_n,ci_fisher,width_fisher,ci_pearson,width_pearson,ci_george,width_george,ci_tippet,width_tippet,ci_stouffer,width_stouffer,ci_wright,width_wright]





def simulation_tau_3(N: list, n: list, tau1: int,tau2: int, tau3: int, alpha:float=0.05,replication: int=100):

    potential_outcome1=np.zeros((N[0],2))
    if tau1>=0:
        for i in range(tau1):
            potential_outcome1[i]=[0,1]
    if tau1<=0:
        for i in range(abs(tau1)):
            potential_outcome1[i]=[1,0]
    temp=np.random.binomial(1,0.5,N[0]-abs(tau1))
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
    temp=np.random.binomial(1,0.5,N[1]-abs(tau2))
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
    temp=np.random.binomial(1,0.5,N[2]-abs(tau3))
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

    total_p_value=Combined_Permute.p_permute(matrix_n,replication)

    ci_fisher=Combined_Permute.ci_combine(total_p_value,Combined_Permute.fisher,alpha)
    width_fisher=ci_fisher[1]-ci_fisher[0]

    ci_pearson=Combined_Permute.ci_combine(total_p_value,Combined_Permute.pearson,alpha)
    width_pearson=ci_pearson[1]-ci_pearson[0]

    ci_george=Combined_Permute.ci_combine(total_p_value,Combined_Permute.george,alpha)
    width_george=ci_george[1]-ci_george[0]

    ci_tippet=Combined_Permute.ci_combine(total_p_value,Combined_Permute.tippett,alpha)
    width_tippet=ci_tippet[1]-ci_tippet[0]

    ci_stouffer=Combined_Permute.ci_combine(total_p_value,Combined_Permute.stouffer,alpha)
    width_stouffer=ci_stouffer[1]-ci_stouffer[0]

    ci_wright=Combined_Permute.ci_wright(total_p_value,Combined_Permute.stouffer,alpha)
    width_wright=ci_wright[1]-ci_wright[0]


    return [matrix_n,ci_fisher,width_fisher,ci_pearson,width_pearson,ci_george,width_george,ci_tippet,width_tippet,ci_stouffer,width_stouffer,ci_wright,width_wright]

def simulation_plot(tau_list: list, file_name: str, alpha:float=0.05,replication: int=100, simulation: int=50):
    if len(tau_list)==2:
        for N in range(24,121,4):
            strata_N=[N//2,N//2]
            strata_n=[N//4,N//4]
            tau1=int(tau_list[0]*N/2)
            tau2=int(tau_list[1]*N/2)
            for i in range(simulation):
                res=simulation_tau_2(strata_N,strata_n,tau1,tau2,alpha,replication)
                with open(file_name,"a") as csvfile:
                    writer=csv.writer(csvfile)
                    writer.writerow(res)
    if len(tau_list)==3:
        for N in range(24,121,6):
            strata_N=[N//3,N//3,N//3]
            strata_n=[N//6,N//6,N//6]
            tau1=int(tau_list[0]*N/3)
            tau2=int(tau_list[1]*N/3)
            tau3=int(tau_list[2]*N/3)
            for i in range(simulation):
                res=simulation_tau_3(strata_N,strata_n,tau1,tau2,tau3,alpha,replication)
                with open(file_name,"a") as csvfile:
                    writer=csv.writer(csvfile)
                    writer.writerow(res)
simulation_plot([0.6,0.6,0.6],r"E:\CollegeLife\UCB first semester\Extended_Li_and_Ding\Paper_Code\csv\tau666comb.csv")