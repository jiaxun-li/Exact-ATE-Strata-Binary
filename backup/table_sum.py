import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import math
import Wald
from time import time


def table(filename:str,simulation:int):

    with open(filename,newline='') as csvfile:
        dict_table={}
        reader=csv.reader(csvfile,delimiter=',')
        for row in reader:
            if len(row)>0:
                n_temp=re.split('\s|\[|\]|\n|\r',row[0])
                k=0
                for i in n_temp:
                    if i!='':
                        k+=1
                k=k//4
                matrix_N=[[]]*k
                for i in range(k):
                    matrix_N[i]=[0,0,0,0]

                s=i=0
                for num in n_temp:
                    if num!='':
                        matrix_N[s][i]+=int(num)
                        i+=1
                        if i==4:
                            s+=1
                            i=0
                N=np.sum(matrix_N)

                n_temp=re.split('\s|\[|\]|\n|\r',row[1])
                matrix_n=[[]]*k
                for i in range(k):
                    matrix_n[i]=[0,0,0,0]

                s=i=0
                for num in n_temp:
                    if num!='':
                        matrix_n[s][i]+=int(num)
                        i+=1
                        if i==4:
                            s+=1
                            i=0
                N_tuple=tuple([sum(matrix_n[i]) for i in range(k)])


                tau=sum(matrix_N[i][1]-matrix_N[i][2] for i in range(k))
                wald_ci=Wald.wald_interval(matrix_n)
                wald_width=(wald_ci[1]-wald_ci[0])
                wald_cov=(tau/N>=wald_ci[0]) and (tau/N<=wald_ci[1])
                wald_time=float(row[3])

                fast_temp=re.split(',|\[|\]',row[4])
                fast_ci=[int(fast_temp[1]),int(fast_temp[2])]
                fast_width=(fast_ci[1]-fast_ci[0])/N
                fast_cov=(tau>=fast_ci[0]) and (tau<=fast_ci[1])
                fast_time=float(row[5])

                ws_temp=re.split(',|\[|\]',row[6])
                ws_ci=[int(ws_temp[1]),int(ws_temp[2])]
                ws_width=(ws_ci[1]-ws_ci[0])/N
                ws_cov=(tau>=ws_ci[0]) and (tau<=ws_ci[1])
                ws_time=float(row[7])

                perm_comb_temp=re.split(',|\[|\]',row[8])
                perm_comb_ci=[int(perm_comb_temp[1]),int(perm_comb_temp[2])]
                perm_comb_width=(perm_comb_ci[1]-perm_comb_ci[0])/N
                perm_comb_cov=(tau>=perm_comb_ci[0]) and (tau<=perm_comb_ci[1])
                perm_comb_time=float(row[9])

                perm_temp=re.split(',|\[|\]',row[10])
                perm_ci=[int(perm_temp[1]),int(perm_temp[2])]
                perm_width=(perm_ci[1]-perm_ci[0])/N
                perm_cov=(tau>=perm_ci[0]) and (tau<=perm_ci[1])
                perm_time=float(row[11])

                if N_tuple not in dict_table:
                    dict_table[N_tuple]={
                        "wald":[0,0,0],
                        "fast":[0,0,0],
                        "ws":[0,0,0],
                        "perm_comb":[0,0,0],
                        "perm":[0,0,0]
                    }


                dict_table[N_tuple]["wald"][0]+=wald_width/(simulation)
                dict_table[N_tuple]["wald"][1]+=wald_cov/(simulation)
                dict_table[N_tuple]["wald"][2]+=wald_time/(simulation)

                dict_table[N_tuple]["fast"][0]+=fast_width/(simulation)
                dict_table[N_tuple]["fast"][1]+=fast_cov/(simulation)
                dict_table[N_tuple]["fast"][2]+=fast_time/(simulation)

                dict_table[N_tuple]["ws"][0]+=ws_width/(simulation)
                dict_table[N_tuple]["ws"][1]+=ws_cov/(simulation)
                dict_table[N_tuple]["ws"][2]+=ws_time/(simulation)

                dict_table[N_tuple]["perm_comb"][0]+=perm_comb_width/(simulation)
                dict_table[N_tuple]["perm_comb"][1]+=perm_comb_cov/(simulation)
                dict_table[N_tuple]["perm_comb"][2]+=perm_comb_time/(simulation)

                dict_table[N_tuple]["perm"][0]+=perm_width/(simulation)
                dict_table[N_tuple]["perm"][1]+=perm_cov/(simulation)
                dict_table[N_tuple]["perm"][2]+=perm_time/(simulation)

    for N_tuple in dict_table:
        for key in dict_table[N_tuple]:
            for i in range(3):
                dict_table[N_tuple][key][i]=round(dict_table[N_tuple][key][i],2)
    return(dict_table)


def table_comb(filename:str,simulation:int):

    with open(filename,newline='') as csvfile:
        dict_table={}
        reader=csv.reader(csvfile,delimiter=',')
        for row in reader:
            if len(row)>0:
                n_temp=re.split('\s|\[|\]|\n|\r',row[0])
                k=0
                for i in n_temp:
                    if i!='':
                        k+=1
                k=k//4
                matrix_N=[[]]*k
                for i in range(k):
                    matrix_N[i]=[0,0,0,0]

                s=i=0
                for num in n_temp:
                    if num!='':
                        matrix_N[s][i]+=int(num)
                        i+=1
                        if i==4:
                            s+=1
                            i=0
                N=np.sum(matrix_N)
                N_tuple=tuple([sum(matrix_N[i]) for i in range(k)])


                tau=sum(matrix_N[i][1]-matrix_N[i][2] for i in range(k))

                fisher_width=int(row[3])/N
                pearson_width=int(row[5])/N
                george_width=int(row[7])/N
                tippet_width=int(row[9])/N
                stouffer_width=int(row[11])/N
                wright_width=int(row[13])/N

                if N_tuple not in dict_table:
                    dict_table[N_tuple]={
                        "fisher":0,
                        "pearson":0,
                        "george":0,
                        "tippet":0,
                        "stouffer":0,
                        "wright":0
                    }


                dict_table[N_tuple]["fisher"]+=fisher_width/(simulation)
                dict_table[N_tuple]["pearson"]+=pearson_width/(simulation)
                dict_table[N_tuple]["george"]+=george_width/(simulation)
                dict_table[N_tuple]["tippet"]+=tippet_width/(simulation)
                dict_table[N_tuple]["stouffer"]+=stouffer_width/(simulation)
                dict_table[N_tuple]["wright"]+=wright_width/(simulation)

    for N_tuple in dict_table:
        for key in dict_table[N_tuple]:
            dict_table[N_tuple][key]=round(dict_table[N_tuple][key],2)
    return(dict_table)

print(table_comb(r"E:\CollegeLife\UCB first semester\Extended_Li_and_Ding\Paper_Code\csv\table_comb2.csv",100))
