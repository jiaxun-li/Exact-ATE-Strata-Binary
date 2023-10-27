import Extended_Permute
import Combined_Permute
import Hypergeo
import numpy as np
import matplotlib.pyplot as plt
import csv
from threading import Thread
from time import time
import Wald
import os


# Supplement material page 10
matrix_n=[
    [8,21,3,22],
    [8,14,2,24]
]

def case_study(matrix_n, file_name: str,alpha:float=0.05,replication:int=5000):
    N=np.sum(matrix_n)
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
    res= [matrix_n,ci_wald,time_wald,ci_fast,time_fast,ci_ws,time_ws,ci_permute_combine,time_permute_comb,ci_permute,time_permute]
    with open(file_name,"a") as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(res)
    return 
case_study(matrix_n,r"E:\CollegeLife\UCB first semester\Extended_Li_and_Ding\Paper_Code\csv\case_study.csv")