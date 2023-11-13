import utils
import laplacian
import display
import torch
import os
import time

start = time.time()
path = os.path.normpath('C:\Ecole\Master_ID3D\Frequentielle\Code\data/bigguy.obj')
# path = os.path.normpath('C:\Ecole/4A\TSI\projet_tsi\data/armadillo_light.off')

v , f = utils.ReadObj(path)
v[:,2] = -v[:,2]
print(f'v shape : {v.shape}, f shape {f.shape}')

L = laplacian.laplacian_cotangent(v,f)
v = L @ v
print('L',L)
# v_recontruction = laplacian.MeshReconstruction(v,f,2000)
# laplacian.svd(v,4)

# ML = laplacian.graph_laplacian(v,f)
# v = laplacian.MeshReconstruction(v,f,10)
# Q = laplacian.graph_laplacian(v,f)
# vfreg = torch.matmul(Q,v)

# vML = torch.matmul(ML, v)
# smooth_v = laplacian.smooth_laplacian(v,f) 
# d_maps = laplacian.diffusion_maps(v,f,alpha=0.9)*1
# laplacian.svd(v,3)
# display.display(smooth_v,f)
# print(f'v reconstruction {v_recontruction.shape}')
display.display(v,f)

end = time.time()
global_time = end-start
print(f'time {global_time//60} min {global_time%60} sec')






