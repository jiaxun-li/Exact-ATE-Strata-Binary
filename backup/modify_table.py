import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import math
import simu_table
from time import time

with open(r"E:\CollegeLife\UCB first semester\Extended_Li_and_Ding\Paper_Code\csv\table1.csv",newline='') as inputfile,open(r"E:\CollegeLife\UCB first semester\Extended_Li_and_Ding\Paper_Code\csv\table1_modified.csv",'a') as outputfile:
    reader=csv.reader(inputfile,delimiter=',')
    writer=csv.writer(outputfile)
    for row in reader:
        if len(row)>0:
            N_temp=re.split('\s|\[|\]|\n|\r',row[0])
            k=0
            for num in N_temp:
                if num!='':
                    k+=1
            k=int(k/4)
            matrix_N=np.zeros((k,4),dtype=int)
            s=i=0
            for num in N_temp:
                if num!='':
                    matrix_N[s][i]+=int(num)
                    i+=1
                    if i==4:
                        s+=1
                        i=0
            N=np.sum(matrix_N)

            matrix_n=np.zeros((k,4),dtype=int)
            n_temp=re.split('\s|\[|\]|\n|\r',row[1])
            s=i=0
            for num in n_temp:
                if num!='':
                    matrix_n[s][i]+=int(num)
                    i+=1
                    if i==4:
                        s+=1
                        i=0
            if np.array_equal(matrix_N,np.array([[1,12,0,1],[2,55,1,2]])):
                new_N=np.array([[2,12,0,1],[2,55,1,2]])
                res=simu_table.simu_matrix_N(new_N,[10,40])
                writer.writerow(res)
            else:
                writer.writerow([matrix_N,matrix_n,row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11]])