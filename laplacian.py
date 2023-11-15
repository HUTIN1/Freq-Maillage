import torch
from functools import cache 
#garde en cache le output de la fonction et le input, car si on redemande le meme input à la fonction le @cache retournera le resultat deja calculé
from tqdm import tqdm

import utils


@cache
def adjancy_matrix(nvertex , f):
    #adjacency matrix
    W = torch.zeros((nvertex,nvertex)).cuda()
    for idx in range(nvertex):
        wy, _ = torch.where(f == idx)
        neig = torch.unique(f[wy,:])
        W[idx,neig] = 1
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
    #symmetrically normalized Laplacian
    frow , fcol = f.shape
    if frow < fcol :
        f = f.T
    nvertex = v.shape[0]
    
    W = adjancy_matrix(nvertex,f)

    D = degree_matrix(nvertex,f)

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

    # eigenValues = eigenValues[idx]
    # eigenVectors = eigenVectors[:,idx]
    d_maps = torch.matmul(D,eigenVectors)
    print(f' dmaps shape {d_maps.shape}')


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
    # Lapla = laplacian_cotangent(v,f)
    L, V = torch.linalg.eig(Lapla)
    
    L = L.type(torch.float32)
    V = V.type(torch.float32)

    invr = [i for i in range(number,0,-1)]
    M = V[:,invr].cuda()

    l = V[:,:number] @ torch.diag(L).cuda()[:number,:number] @ V[:,:number].t()
    vp = Lapla @ v 
    v_reconstruction = torch.linalg.pinv(l) @ vp

    return v_reconstruction



def laplacian_cotangent(v,f):
    """
    Laplace–Beltrami (Intrinsic Laplacian)

    Lij = | 1/2 (cot(alpha) + cot(beta)) if {i,j} in E (edge)
          | - sum_(i dif j) Lij          if i=j
          | 0                            otherwise
                r = row
                   /|\
                 /  |  \
            qr /    |    \  rt
             /      |      \
           /        |        \
q = d[0] |alpha  rs |   beta   | t = d[1]
           \        |        /
             \      |      /
           qs  \    |    /  st
                 \  |  / 
                   \|/
                  s = col
    rs**2 = qr**2  + qs**2- 2.qs.qr.cos(alpha)
    """
    

    nbvertices = v.shape[0]
    W = adjancy_matrix(nbvertices,f)
    L = torch.zeros((nbvertices,nbvertices)).cuda()
    pbar = tqdm(desc="Compute Laplace–Beltrami",total=nbvertices**2/2-nbvertices)
    for row in range(nbvertices):


        for col in range(row+1,nbvertices):
            if W[row,col] == 1 :
                # print('row',row,'col',col)
                r = v[row,:]
                s = v[col,:]
                
                # b, b2 = torch.where(f == row)
                # c , c2 = torch.where(f == col)
                # d ,d2 = torch.where(b.unsqueeze(0).expand(c.shape[0],-1) == c.unsqueeze(-1))
                b = torch.argwhere(W[row,:]).squeeze()
                c = torch.argwhere(W[col,:]).squeeze()


                d, d2 = torch.where(b.unsqueeze(1).expand(-1,c.shape[0]) == c)

                
                q = v[b[d[0]],:]
                t = v[b[d[1]],:]

                # print('f',f)
                # print('qidx',qidx,'tidx',tidx)
                # quit()
                

                rs = torch.dist(r,s)
                qr = torch.dist(q,r)
                qs = torch.dist(q,s)
                rt = torch.dist(t,r)
                st = torch.dist(t,s)

                wcos = (qr**2 + qs**2 - rs)/(2*qr*qs)
                wsin = torch.sqrt(1 - wcos**2)
                wcot = wcos / wsin

                zcos = (rt**2 + st**2 - rs)/(2*rt*st)
                zsin = torch.sqrt(1 - zcos**2)
                zcot = zcos / zsin

                L[row,col] = 1/2*(wcot + zcot)
                L[col,row] = L[row,col]

            pbar.update(1)

    l = range(nbvertices)
    L[l,l] = -torch.sum(L,None)

    # L = torch.nn.functional.normalize(L)


    return L


