import plotly.graph_objects as go
import torch
import numpy as np




# U, S , Vh = np.linalg.svd(matrix_vertex.T,full_matrices=False)

def display(v , f,color=None):
    
    if v.shape[0] != 3:
        v = v.t()
    if f.shape[0] !=3 :
        f = f.t()

    v = v.cpu().numpy()
    f = f.cpu().numpy()
    if color is not None:
        mincolor , _= torch.min(color,axis=0)
        maxcolor , _ = torch.max(color,axis=0)
        color = (color + mincolor.unsqueeze(0)*-1)/ (maxcolor.unsqueeze(0) + torch.abs(mincolor.unsqueeze(0))) *255
        color = color.cpu().tolist()
        # print(color)
        # color = np.ones((8,3))
        # colorstr= []
        # for i in range(8):
        #     colorstr.append(f'rgb({color[i,0]},{color[i,1]},{color[i,2]})')


    fig = go.Figure(data=
            go.Mesh3d(
                x=v[0,:],
                y=v[1,:],
                z=v[2,:],
                i = f[0,:],
                j = f[1,:],
                k = f[2,:],
                vertexcolor=color
                ))

    fig.show()
