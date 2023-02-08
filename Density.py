import math
import random
import numpy as np
from numpy import linalg as LA
import sympy as sp
import matplotlib.pyplot as plt
import networkx as nx

N = 100
m =10
p=[0.05, 0.1, 0.2, 0.5]
NUMITER = 10


def funDelt (a,b):
    if a == b:
        return 1
    else:
        return 0
x = np.arange(-15, 15, 0.01)
x = x.round(1)

for pit in range (len(p)):
    rho = [0] * len(x)
    for iter in range(NUMITER):

        kmean = 0
        print(iter)
        G = nx.erdos_renyi_graph(N, p[pit], seed=None, directed=False)
        A = nx.adjacency_matrix(G)
        Matrix = [[0 for j in range(N)] for i in range(N)]

        for i in range (N):
            for j in range (N):
                Matrix[i][j] = A[i,j]

        eg = LA.eigvals(Matrix)
        eg = eg.round(1)

        for i in range (len(x)):
            a = 0
            for j in range (len(eg)):
                a +=(funDelt(x[i], eg[j]))/N
            rho[i] += a


    #numrho = [x*math.sqrt((N*p[pit]*(1-p[pit]))) for x in rho]
    #numx = [x/math.sqrt((N*p[pit]*(1-p[pit]))) for x in x]

        k = G.degree

        for i in range (N):
            kmean += k[i]

        kmean = kmean / N

    rhokmean = [x*math.sqrt(kmean) for x in rho]
    xkmean = [x / math.sqrt(kmean) for x in x]


    plt.figure(1)
    plt.plot(xkmean,rhokmean, label = p[pit])
    plt.xlabel("λ/<k>^1/2")
    plt.ylabel("ρ(λ)<k>^1/2")
    plt.legend()
plt.show()

