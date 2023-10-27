import Combined_Permute
import numpy as np
import csv

# matrix_N_list=[
#     np.array([[10,10,10,10],[10,10,10,10]]),
#     np.array([[3,8,4,5],[0,19,1,0]]),
#     np.array([[3,23,2,2],[4,2,30,4]]),
#     np.array([[2,24,0,4],[1,26,2,1]]),
#     np.array([[1,0,9,0],[0,40,0,0]]),
#     np.array([[5,5,5,5],[20,50,2,8]]),
#     np.array([[2,12,0,1],[2,55,1,2]]),
#     np.array([[2,2,12,4],[3,64,1,2]]),
#     np.array([[0,16,0,4],[3,9,1,7],[5,5,5,5]]),
#     np.array([[1,13,1,0],[0,18,0,2],[0,20,0,5]]),
#     np.array([[0,19,1,0],[3,4,4,4],[0,2,18,0]]),
#     np.array([[3,2,2,3],[4,0,10,6],[5,23,3,9]])
# ]
# n_list=[
#     [10,10],
#     [15,15],
#     [5,30],
#     [5,25],
#     [5,20],
#     [15,60],
#     [10,40],
#     [5,60],
#     [5,10,15],
#     [10,10,10],
#     [5,5,5],
#     [5,5,30]
# ]

matrix_N_list=[
    np.array([[5,0,0,45],[10,0,0,40],[10,0,0,40],[5,0,0,75]])
]
n_list=[
    [25,25,25,40]
]



def simu_matrix_N(matrix_N: list, n: list, alpha:float=0.05,replication:int=10000):
    k=len(matrix_N)
    matrix_n=np.zeros((k,4),dtype=int)
    N=np.sum(matrix_N)
    for i in range(k):
        N11,N10,N01,N00=matrix_N[i]
        Ni=sum(matrix_N[i])
        
        N11_t=N10_t=N01_t=N00_t=0
        experiment=[0]*Ni
        experiment[:n[i]]=[1]*n[i]
        np.random.shuffle(experiment)
        
        N11_t=sum(experiment[:N11])
        N10_t=sum(experiment[N11:N11+N10])
        N01_t=sum(experiment[N11+N10:N11+N10+N01])
        N00_t=sum(experiment[N11+N10+N01:])

        matrix_n[i][0]=N10_t+N11_t
        matrix_n[i][1]=N00_t+N01_t
        matrix_n[i][2]=N01-N01_t+N11-N11_t
        matrix_n[i][3]=N00-N00_t+N10-N10_t

    total_p_value=Combined_Permute.p_permute(matrix_n,replication)

    ci_fisher=Combined_Permute.ci_combine(total_p_value,Combined_Permute.fisher)
    width_fisher=ci_fisher[1]-ci_fisher[0]

    ci_pearson=Combined_Permute.ci_combine(total_p_value,Combined_Permute.pearson)
    width_pearson=ci_pearson[1]-ci_pearson[0]

    ci_george=Combined_Permute.ci_combine(total_p_value,Combined_Permute.george)
    width_george=ci_george[1]-ci_george[0]

    ci_tippet=Combined_Permute.ci_combine(total_p_value,Combined_Permute.tippett)
    width_tippet=ci_tippet[1]-ci_tippet[0]

    ci_stouffer=Combined_Permute.ci_combine(total_p_value,Combined_Permute.stouffer)
    width_stouffer=ci_stouffer[1]-ci_stouffer[0]

    ci_wright=Combined_Permute.ci_wright(total_p_value)
    width_wright=ci_wright[1]-ci_wright[0]

    return [matrix_N,matrix_n,ci_fisher,width_fisher,ci_pearson,width_pearson,ci_george,width_george,ci_tippet,width_tippet,ci_stouffer,width_stouffer,ci_wright,width_wright]

def simu_table(matrix_N_list: list,n_list: list, file_name: str,alpha:float=0.05,replication:int=100,simulation:int=100):
    L=len(matrix_N_list)
    for i in range(L):
        for j in range(simulation):
            res=simu_matrix_N(matrix_N_list[i],n_list[i],alpha,replication)
            with open(file_name,"a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(res)
simu_table(matrix_N_list,n_list,r"E:\CollegeLife\UCB first semester\Extended_Li_and_Ding\Paper_Code\csv\table_comb2.csv")