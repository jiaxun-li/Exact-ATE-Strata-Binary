import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import math

simulation=100
k=2
tau=0.9

def wald_interval(matrix_n: list,alpha: float=0.05):
    k=len(matrix_n)
    N=[sum(matrix_n[i]) for i in range(k)]
    n=[matrix_n[i][0]+matrix_n[i][1] for i in range(k)]
    pai=[N[i]/np.sum(N) for i in range(k)]
    tau_hat=sum(pai[i]*(matrix_n[i][0]/n[i]-matrix_n[i][2]/(N[i]-n[i])) for i in range(k))
    S1_2=[1/(n[i]-1)*(matrix_n[i][0]*(1-matrix_n[i][0]/n[i])**2+matrix_n[i][1]*(matrix_n[i][0]/n[i])**2)for i in range(k)]
    S0_2=[1/(N[i]-n[i]-1)*(matrix_n[i][2]*(1-matrix_n[i][2]/(N[i]-n[i]))**2+matrix_n[i][3]*(matrix_n[i][2]/(N[i]-n[i]))**2)for i in range(k)]
    Vs=sum(pai[i]*(S1_2[i]/n[i]+S0_2[i]/(N[i]-n[i])) for i in range(k))
    return [tau_hat-scipy.stats.norm.ppf(1-alpha/2)*math.sqrt(Vs),tau_hat+scipy.stats.norm.ppf(1-alpha/2)*math.sqrt(Vs)]


with open('tau9BE0.csv',newline='') as csvfile:
    dict_fast={}
    dict_ws={}
    dict_perm_comb={}
    dict_perm={}
    dict_wald={}
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

            fast_temp=re.split(',|\[|\]',row[1])
            fast_ci=[int(fast_temp[1]),int(fast_temp[2])]
            fast_cov=fast_ci[0]<=tau*N and fast_ci[1]>=tau*N

            ws_temp=re.split(',|\[|\]',row[2])
            ws_ci=[int(ws_temp[1]),int(ws_temp[2])]
            ws_cov=ws_ci[0]<=tau*N and ws_ci[1]>=tau*N

            perm_comb_temp=re.split(',|\[|\]',row[3])
            perm_comb_ci=[int(perm_comb_temp[1]),int(perm_comb_temp[2])]
            perm_comb_cov=perm_comb_ci[0]<=tau*N and perm_comb_ci[1]>=tau*N


            perm_temp=re.split(',|\[|\]',row[4])
            perm_ci=[int(perm_temp[1]),int(perm_temp[2])]
            perm_cov=perm_ci[0]<=tau*N and perm_ci[1]>=tau*N

            wald_cov=wald_interval(matrix_n)[1]>=tau and wald_interval(matrix_n)[0]<=tau

            if N not in dict_fast.keys():
                dict_fast[N]=0
                dict_ws[N]=0
                dict_perm[N]=0
                dict_perm_comb[N]=0
                dict_wald[N]=0


            dict_fast[N]+=fast_cov/(simulation)
            dict_ws[N]+=ws_cov/(simulation)
            dict_perm_comb[N]+=perm_comb_cov/(simulation)
            dict_perm[N]+=perm_cov/(simulation)
            dict_wald[N]+=wald_cov/simulation

plt.plot(dict_fast.keys(),dict_fast.values(),label='fast')
plt.plot(dict_ws.keys(),dict_ws.values(),label='ws')
plt.plot(dict_perm_comb.keys(),dict_perm_comb.values(),label='perm_comb')
plt.plot(dict_perm.keys(),dict_perm.values(),label='perm')
plt.plot(dict_wald.keys(),dict_wald.values(),label="wald")
plt.title("tau=0,BE=1.6")
plt.xlabel("N")
plt.ylabel("ci_width")
plt.legend()
plt.show()
        