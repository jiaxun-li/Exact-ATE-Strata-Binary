import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import math
import Extended_Permute_balanced
from time import time
k=2


with open(r"E:\CollegeLife\UCB first semester\Extended_Li_and_Ding\Paper_Code\csv\tau-44.csv",newline='') as inputfile,open(r"E:\CollegeLife\UCB first semester\Extended_Li_and_Ding\Paper_Code\csv\tau-44_modified.csv",'a') as outputfile:
    reader=csv.reader(inputfile,delimiter=',')
    writer=csv.writer(outputfile)
    for row in reader:
        if len(row)>0:
            matrix_n=np.zeros((2,4),dtype=int)
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
            s=time()
            ci=Extended_Permute_balanced.confidence_interval_permute(matrix_n)
            e=time()
            t=round(e-s,5)
            writer.writerow([matrix_n,row[1],row[2],row[3],ci,row[5],row[6],row[7],t])