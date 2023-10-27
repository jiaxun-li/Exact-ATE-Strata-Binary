import numpy as np
import scipy.stats
import math
def wald_interval(matrix_n: list,alpha: float=0.05):
    k=len(matrix_n)
    N=[sum(matrix_n[i]) for i in range(k)]
    n=[matrix_n[i][0]+matrix_n[i][1] for i in range(k)]
    pai=[N[i]/np.sum(N) for i in range(k)]
    tau_hat=sum(pai[i]*(matrix_n[i][0]/n[i]-matrix_n[i][2]/(N[i]-n[i])) for i in range(k))
    S1_2=[1/(n[i]-1)*(matrix_n[i][0]*(1-matrix_n[i][0]/n[i])**2+matrix_n[i][1]*(matrix_n[i][0]/n[i])**2) for i in range(k)]
    S0_2=[ 1/(N[i]-n[i]-1)*(matrix_n[i][2]*(1-matrix_n[i][2]/(N[i]-n[i]))**2+matrix_n[i][3]*(matrix_n[i][2]/(N[i]-n[i]))**2)for i in range(k)]
    Vs=sum(pai[i]**2*(S1_2[i]/n[i]+S0_2[i]/(N[i]-n[i])) for i in range(k))
    return [tau_hat-scipy.stats.norm.ppf(1-alpha/2)*math.sqrt(Vs),tau_hat+scipy.stats.norm.ppf(1-alpha/2)*math.sqrt(Vs)]
