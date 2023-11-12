import utils
import laplacian
import display
import torch
import os


path = os.path.normpath('C:\Ecole\Master_ID3D\Frequentielle\Code\data/dino.obj')
# path = os.path.normpath('C:\Ecole/4A\TSI\projet_tsi\data/armadillo_light.off')

v , f = utils.ReadObj(path)
print(f'v shape : {v.shape}, f shape {f.shape}')
# v_recontruction = laplacian.MeshReconstruction(v,f,2000)
# laplacian.svd(v,4)

# ML = laplacian.graph_laplacian(v,f)

# vML = torch.matmul(ML, v)
# smooth_v = laplacian.smooth_laplacian(v,f) 
# d_maps = laplacian.diffusion_maps(v,f,alpha=0.9)*1
# laplacian.svd(v,3)
# display.display(smooth_v,f)
# print(f'v reconstruction {v_recontruction.shape}')
display.display(v,f)






