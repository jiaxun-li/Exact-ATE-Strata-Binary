import numpy as np
import math
def t(w):
    m = w[0] + w[1]
    n = np.sum(w)
    if m == 0:
        return - w[2]/(w[2] + w[3])
    elif m==n:
        return w[0]/(w[0] + w[1]) 
    else:
        return (w[0]/(w[0] + w[1])) - (w[2]/(w[2] + w[3]))
    
def findp(N, n, l=int(1e4),seed=80):
    np.random.seed(seed)
    obs = t(n)
    tN = (N[1] - N[2])/sum(N)
    N11, N10, N01, N00 = np.cumsum(N)
    k = n[0]+n[1]
    cnt = 0
    for i in range(l):
        treat = np.random.choice(N00, k, replace=False)
        assert np.max(treat)<N00
        assert len(np.unique(treat)) == k
        count = [np.sum(treat<N11), np.sum((treat<N10)&(treat>=N11)), 
                 np.sum((treat<N01)&(treat>=N10)), np.sum(treat>=N01)]
        ## The original four formula about n11,...,n00 is not true. Since the paper uses a different notation
        ## about N10 and N01 from the lecture notes, I think maybe you made a mistake about that.
        n11 = count[0]*1 + count[1]*1 + count[2]*0 + count[3]*0
        n10 = count[0]*0 + count[1]*0 + count[2]*1 + count[3]*1
        n01 = (N[0] - count[0])*1 + (N[1]-count[1])*0 + (N[2]-count[2])*1 + (N[3]-count[3])*0
        n00 = (N[0] - count[0])*0 + (N[1]-count[1])*1 + (N[2]-count[2])*0 + (N[3]-count[3])*1
        tn = t([n11, n10, n01, n00])
        if round(abs(tn-tN),10) >= round(abs(obs - tN),10):
            cnt += 1
    return (cnt/l) 

def permutation_test(n, alpha, Tn):
    N = sum(n)
    n11, n10, n01, n00 = n
    cnt = 0
    for j in range(N+1):
        if j >= int(N*Tn)+n01 and j >= n11 and N >= j+n10 and n11+int(N*Tn)+n10+n01 >= j:
            lower_bound = max(0, int(N*Tn), j-n11-n01, n11+n01+int(N*Tn)-j)
            upper_bound = min(j, n11+n00, n10+n01+int(N*Tn), N+int(N*Tn)-j)
            v10=lower_bound
            v = (j-v10, v10, v10-int(N*Tn), N-j-v10+int(N*Tn))
            p_value = findp(v, n)
            if Tn<=-62 and p_value>=alpha:
                print(v,p_value)
            cnt += 1
            if p_value >= alpha:
                   return 0,cnt
            elif v[1] == 0 and v[2] == 0:
                p_value = findp([v[0]-1, v[1]+1, v[2]+1, v[3]-1], n)
                cnt += 1
                if p_value >= alpha:
                    return [0,cnt]
    return [1,cnt]

def binary_search(k1, k2, f):
    a = k1
    b = k2
    cnt=0
    while b > a + 1:
        c = math.floor((a + b) / 2)
        f_c=f(c)
        if f_c[0] == 0:
            a = c
            cnt+=f_c[1]
        else:
            b = c
            cnt+=f_c[1]
    if a == k1:
        if f(k1)[0] == 0:
            return [k1,cnt]
        else:
            return [k1 - 1,cnt]
    elif b == k2:
        if f(k2)[0] == 0:
            return [k2,cnt]
        else:
            return k2 - 1,cnt
    else:
        return a,cnt
    
def binary_search_opp(k1, k2, f):
    a = k1
    b = k2
    cnt=0
    while b > a + 1:
        c = math.ceil((a + b) / 2)
        f_c=f(c)
        ## The original stuff is: a=c  if f(c)==0, else b=c 
        if f_c[0] == 0:
            b = c
            cnt+=f_c[1]
        else:
            a = c
            cnt+=f_c[1]

    if b == k2:
        if f(k2)[0] == 0:
            return k2,cnt
        else:
            return k2 + 1,cnt
    elif a == k1:
        if f(k1)[0] == 0:
            return k1,cnt 
        else:
            return k1 + 1,cnt
    else:
        return b,cnt

def find_interval(alpha, n):
    N = sum(n)
    n11, n10, n01, n00 = n
    Tn = t(n)
    def f(x):
        return permutation_test(n, alpha, x/N)
    
    k2 = round(N*Tn)
    k1 = n11+n00-N 

    n11+n00
    L,cnt1 = binary_search_opp(k1, k2, f)
    
    k1 = round(N*Tn)
    k2 = n00 + n11 #the original stuff is n01 + n10

    U,cnt2 = binary_search(k1, k2, f)
    #U=0
    return np.array([L, U]),cnt1+cnt2

print(find_interval(0.05, [32, 68, 51, 49]))