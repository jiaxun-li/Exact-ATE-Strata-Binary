import Wald
import numpy as np
import scipy.stats
import math
import matplotlib.pyplot as plt


def wald_simu(matrix_N,n_list,simu=5000):
    S=0
    l=0
    for i in range(simu):
        k=len(matrix_N)
        matrix_n=np.zeros((k,4),dtype=int)
        N=np.sum(matrix_N)
        for i in range(k):
            N11,N10,N01,N00=matrix_N[i]
            Ni=sum(matrix_N[i])

            N11_t=N10_t=N01_t=N00_t=0
            experiment=[0]*Ni
            experiment[:n_list[i]]=[1]*n_list[i]
            np.random.shuffle(experiment)

            N11_t=sum(experiment[:N11])
            N10_t=sum(experiment[N11:N11+N10])
            N01_t=sum(experiment[N11+N10:N11+N10+N01])
            N00_t=sum(experiment[N11+N10+N01:])

            matrix_n[i][0]=N10_t+N11_t
            matrix_n[i][1]=N00_t+N01_t
            matrix_n[i][2]=N01-N01_t+N11-N11_t
            matrix_n[i][3]=N00-N00_t+N10-N10_t
        a=Wald.wald_interval(matrix_n)
        l+=(a[1]-a[0])
        tau=sum(matrix_N[i][1]-matrix_N[i][2] for i in range(k))/N
        S+=(a[0]<=tau)&(a[1]>=tau)
    return(l/simu,S/simu)

def wald_plot(low=48,up=400,rep=100,k=4):
    l=(up-low)//2+1
    x=range(low,up+1,2)
    y=[0]*l
    y2=[0.95]*l
    for j in range(rep):
        for i in range(low,up+1,2):
            matrix_N=[]
            for l in range(2):
                matrix_N.append([i//4,0,0,i-i//4])
            for l in range(2,k):
                matrix_N.append([0,0,0,i])
            n_list=[i//4]*k
            l,s=wald_simu(matrix_N,n_list)
            y[(i-low)//2]+=s/rep
    plt.plot(x,y)
    plt.plot(x,y2,linestyle='--')
    plt.xlabel("N")
    plt.title("k=4, rare event, only two strata has event")
    plt.ylabel("coverage probability")
    plt.show()
wald_plot()