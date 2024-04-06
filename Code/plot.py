# Plot the figures in Section 5.1

import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import math
from time import time

def wald_interval(matrix_n: list,alpha: float=0.05):
    k=len(matrix_n)
    N=[sum(matrix_n[i]) for i in range(k)]
    n=[matrix_n[i][0]+matrix_n[i][1] for i in range(k)]
    pai=[N[i]/np.sum(N) for i in range(k)]
    tau_hat=sum(pai[i]*(matrix_n[i][0]/n[i]-matrix_n[i][2]/(N[i]-n[i])) for i in range(k))
    S1_2=[1/(n[i]-1)*(matrix_n[i][0]*(1-matrix_n[i][0]/n[i])**2+matrix_n[i][1]*(matrix_n[i][0]/n[i])**2)for i in range(k)]
    S0_2=[1/(N[i]-n[i]-1)*(matrix_n[i][2]*(1-matrix_n[i][2]/(N[i]-n[i]))**2+matrix_n[i][3]*(matrix_n[i][2]/(N[i]-n[i]))**2)for i in range(k)]
    Vs=sum(pai[i]**2*(S1_2[i]/n[i]+S0_2[i]/(N[i]-n[i])) for i in range(k))
    return [tau_hat-scipy.stats.norm.ppf(1-alpha/2)*math.sqrt(Vs),tau_hat+scipy.stats.norm.ppf(1-alpha/2)*math.sqrt(Vs)]

def plot_ci(filename:str,title:str,k:int,simulation:int):

    with open(filename,newline='') as csvfile:
        dict_fast={}
        dict_ws={}
        dict_perm_comb={}
        dict_perm={}
        dict_wald={}
        dict_perm_nostrata={}
        reader=csv.reader(csvfile,delimiter=',')
        for row in reader:
            if len(row)>0:
                matrix_n=[[]]*k
                for i in range(k):
                    matrix_n[i]=[0,0,0,0]
                n_temp=re.split('\s|\[|\]|\n|\r',row[0])
                s=i=0
                for num in n_temp:
                    if num!='':
                        matrix_n[s][i]+=int(num)
                        i+=1
                        if i==4:
                            s+=1
                            i=0
                N=np.sum(matrix_n)
                matrix_n_nostrata=[matrix_n[0][i]+matrix_n[1][i] for i in range(4)]

                fast_temp=re.split(',|\[|\]',row[1])
                fast_ci=[int(fast_temp[1]),int(fast_temp[2])]
                fast_width=(fast_ci[1]-fast_ci[0])/N

                if row[2]!='':
                    ws_temp=re.split(',|\[|\]',row[2])
                    ws_ci=[int(ws_temp[1]),int(ws_temp[2])]
                    ws_width=(ws_ci[1]-ws_ci[0])/N

                perm_comb_temp=re.split(',|\[|\]',row[3])
                perm_comb_ci=[int(perm_comb_temp[1]),int(perm_comb_temp[2])]
                perm_comb_width=(perm_comb_ci[1]-perm_comb_ci[0])/N

                perm_temp=re.split(',|\[|\]',row[4])
                perm_ci=[int(perm_temp[1]),int(perm_temp[2])]
                perm_width=(perm_ci[1]-perm_ci[0])/N


                wald_width=wald_interval(matrix_n)[1]-wald_interval(matrix_n)[0]
                if N not in dict_fast.keys():
                    dict_fast[N]=0
                    if row[2]!='':
                        dict_ws[N]=0
                    dict_perm[N]=0
                    dict_perm_comb[N]=0
                    dict_wald[N]=0


                dict_fast[N]+=fast_width/(simulation)
                if row[2]!='':
                    dict_ws[N]+=ws_width/(simulation)
                dict_perm_comb[N]+=perm_comb_width/(simulation)
                dict_perm[N]+=perm_width/(simulation)
                dict_wald[N]+=wald_width/simulation

    plt.plot(dict_fast.keys(),dict_fast.values(),label='fast',color="black",linestyle='-.')
    plt.plot(dict_ws.keys(),dict_ws.values(),label='ws',color="black",linestyle=':')
    plt.plot(dict_perm_comb.keys(),dict_perm_comb.values(),label='perm_comb',color="black",linestyle='--')
    plt.plot(dict_perm.keys(),dict_perm.values(),label='perm',color="black",marker='o',markersize=2)
    plt.plot(dict_wald.keys(),dict_wald.values(),label="wald",color="black")
    plt.title(title)
    plt.xlabel("N")
    plt.ylabel("ci_width")
    plt.legend()
    plt.show()


def plot_time(filename:str,title:str,k:int,simulation:int):

    with open(filename,newline='') as csvfile:
        dict_fast={}
        dict_ws={}
        dict_perm_comb={}
        dict_perm={}
        dict_wald={}
        dict_perm_nostrata={}
        reader=csv.reader(csvfile,delimiter=',')
        for row in reader:
            if len(row)>0:
                matrix_n=[[]]*k
                for i in range(k):
                    matrix_n[i]=[0,0,0,0]
                n_temp=re.split('\s|\[|\]|\n|\r',row[0])
                s=i=0
                for num in n_temp:
                    if num!='':
                        matrix_n[s][i]+=int(num)
                        i+=1
                        if i==4:
                            s+=1
                            i=0
                N=np.sum(matrix_n)
                matrix_n_nostrata=[matrix_n[0][i]+matrix_n[1][i] for i in range(4)]

                fast_time=float(row[5])

                if row[2]!='':
                    ws_time=float(row[6])

                perm_comb_time=float(row[7])

                perm_time=float(row[8])

                s=time()
                temp=wald_interval(matrix_n)
                e=time()
                wald_time=e-s
                if N not in dict_fast.keys():
                    dict_fast[N]=0
                    if row[2]!='':
                        dict_ws[N]=0
                    dict_perm[N]=0
                    dict_perm_comb[N]=0
                    dict_wald[N]=0


                dict_fast[N]+=fast_time/(simulation)
                if row[2]!='':
                    dict_ws[N]+=ws_time/(simulation)
                dict_perm_comb[N]+=perm_comb_time/(simulation)
                dict_perm[N]+=perm_time/(simulation)
                dict_wald[N]+=wald_time/simulation

    plt.plot(dict_fast.keys(),dict_fast.values(),label='fast',color="black",linestyle='-.')
    plt.plot(dict_ws.keys(),dict_ws.values(),label='ws',color="black",linestyle=':')
    plt.plot(dict_perm_comb.keys(),dict_perm_comb.values(),label='perm_comb',color="black",linestyle='--')
    plt.plot(dict_perm.keys(),dict_perm.values(),label='perm',color="black",marker='o',markersize=2)
    plt.plot(dict_wald.keys(),dict_wald.values(),label="wald",color="black")
    plt.title(title)
    plt.xlabel("N")
    plt.ylabel("time(s)")
    plt.legend()
    plt.show()

def plot_ci_combine(filename:str,title:str,k:int,simulation:int):

    with open(filename,newline='') as csvfile:
        dict_fisher={}
        dict_pearson={}
        dict_george={}
        dict_tippet={}
        dict_stouffer={}
        reader=csv.reader(csvfile,delimiter=',')
        for row in reader:
            if len(row)>0:
                matrix_n=[[]]*k
                for i in range(k):
                    matrix_n[i]=[0,0,0,0]
                n_temp=re.split('\s|\[|\]|\n|\r',row[0])
                s=i=0
                for num in n_temp:
                    if num!='':
                        matrix_n[s][i]+=int(num)
                        i+=1
                        if i==4:
                            s+=1
                            i=0
                N=np.sum(matrix_n)

                fisher_width=int(row[2])/N
                pearson_width=int(row[4])/N
                george_width=int(row[6])/N
                tippet_width=int(row[8])/N
                stouffer_width=int(row[10])/N


                if N not in dict_fisher.keys():
                    dict_fisher[N]=0
                    dict_pearson[N]=0
                    dict_george[N]=0
                    dict_tippet[N]=0
                    dict_stouffer[N]=0


                dict_fisher[N]+=fisher_width/(simulation)
                dict_pearson[N]+=pearson_width/(simulation)
                dict_george[N]+=george_width/(simulation)
                dict_tippet[N]+=tippet_width/(simulation)
                dict_stouffer[N]+=stouffer_width/simulation

    plt.plot(dict_fisher.keys(),dict_fisher.values(),label='fisher',color="black",linestyle='-.')
    plt.plot(dict_pearson.keys(),dict_pearson.values(),label='pearson',color="black",linestyle=':')
    plt.plot(dict_george.keys(),dict_george.values(),label='george',color="black",linestyle='--')
    plt.plot(dict_tippet.keys(),dict_tippet.values(),label='tippet',color="black",marker='o',markersize=2)
    plt.plot(dict_stouffer.keys(),dict_stouffer.values(),label="stouffer",color="black")
    plt.title(title)
    plt.xlabel("N")
    plt.ylabel("ci_width")
    plt.legend()
    plt.show()

