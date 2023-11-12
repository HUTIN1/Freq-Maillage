import numpy as np
import torch
from functools import cache
import utils

@cache
def adjancy_matrix(nvertex , f):
    #adjacency matrix
    W = torch.zeros((nvertex,nvertex)).cuda()
    for y in range(nvertex):
        wy, _ = torch.where(f == y)
        neig = torch.unique(f[wy,:])
        W[y,neig] = 1
    W -= torch.eye(nvertex).cuda()
    return W

@cache
def degree_matrix(nvertex,f):
    #degree matrix
    D = torch.eye(nvertex).cuda().type(dtype=torch.int64)
    unique , counts = torch.unique(f.view(3*f.shape[0]),return_counts=True)
    print(f'unique dtype {unique.dtype}, counts dtype {counts.dtype}')
    D[unique,unique] = counts
    return D

def graph_laplacian(v,f,alpha=0.5):
    frow , fcol = f.shape
    if frow < fcol :
        f = f.T
    nvertex = v.shape[0]
    
    W = adjancy_matrix(nvertex,f)

    D = degree_matrix(nvertex,f)

    # Dsqrt = D ** (-alpha)
    # #pass all the inf value to 0
    # x, y = torch.where(torch.isinf(Dsqrt))
    # Dsqrt[x,y] = 0
    # Dsqrt = Dsqrt.type(dtype = torch.float32)
    Dsqrt = utils.inv_pow(D,alpha)

    #graph laplacian
    K = D.type(dtype=torch.int64) - W.type(dtype=torch.int64)
    K = K.type(dtype = torch.float32)

    #Normalized Graph Laplacian
    Q =torch.matmul(Dsqrt, torch.matmul(K, Dsqrt))
    return Q



def smooth_laplacian(v,f):
    frow , fcol = f.shape
    if frow < fcol :
        f = f.T
    nvertex = v.shape[0]
    
    W = adjancy_matrix(nvertex,f)
    new_v = torch.zeros((nvertex,3)).cuda()
    for i in range(nvertex):
        a = torch.argwhere(W[i,:]==1).squeeze()
        new_v[i,:] += torch.sum(v[a,:],axis=0)/(a.shape[0])

    return new_v


#https://en.wikipedia.org/wiki/Diffusion_map
#https://www.kaggle.com/code/rahulrajpl/diffusion-map-for-manifold-learning
def diffusion_maps(v,f,alpha=0.5):
    nvertex = v.shape[0]

    L = graph_laplacian(v,f,alpha=alpha)
    D = utils.inv_pow(degree_matrix(nvertex,f),alpha)
    M = torch.matmul(D,L)
    # M=L
    eigenValues , eigenVectors = torch.linalg.eigh(M)
    idx = eigenValues.argsort(descending= True)

    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    d_maps = torch.matmul(D,eigenVectors)

    return d_maps[:,:3]



def svd(v,number):
    U, S , Vh = torch.linalg.svd(v,full_matrices=True)
    print(f'U shape {U.shape}, S Shape {S.shape}, Vh shape {Vh.shape}')
    S = torch.concatenate((torch.diag(S),torch.zeros((U.shape[0]-S.shape[0],S.shape[0])).cuda()),axis=0)
    V = torch.zeros(v.shape).cuda()
    for i in range(number):
        V += U[:,1].unsqueeze(1)
        print(U[:,1].unsqueeze(1))
        quit()
    print(f'V shape {V.shape}, \n {V}')
    print(f'S \n {S}')




#the first eig value of laplacian graph is 0
def MeshReconstruction(v,f,number):
    Lapla = graph_laplacian(v,f)
    L, V = torch.linalg.eig(Lapla)
    
    L = L.type(torch.float32)
    V = V.type(torch.float32)
    idx = L.argsort(descending = True)
    # L = L[idx]
    # V = V[:,idx]
    print(f'eig value {L}')
    invr = [i for i in range(number,0,-1)]
    M = V[:,invr].cuda()
    # l = M @ torch.diag(L).cuda()[:number,:number] @ M.t()
    l = V[:,:number] @ torch.diag(L).cuda()[:number,:number] @ V[:,:number].t()
    vp = Lapla @ v #l 5x8 v 8x3 = 5x3
    # v_reconstruction = vp.t() @ torch.linalg.pinv(l) #vp.t 3x5 linv 8x5 
    v_reconstruction = torch.linalg.pinv(l) @ vp
    # v_reconstruction = torch.linalg.pinv(Lapla) @ vp
    # print(f'Lapla shape {Lapla.shape}')
    # print(f'v shape {v.shape}')
    # print(f' vp shape {vp.shape}')
    # print(f'vp.t() shape {vp.t().shape}')
    # print(f'l shape {l.shape}')
    # print(f'l pinv shape {torch.linalg.pinv(l).shape}')
    # print(f'v_reconstruction shape {v_reconstruction.shape}')
    return v_reconstruction