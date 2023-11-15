import utils
import laplacian
import display
import os
import time

start = time.time()
path = os.path.normpath('C:\Ecole\Master_ID3D\Frequentielle\Freq-Maillage\data/robot.obj')
# path = os.path.normpath('C:\Ecole\Master_ID3D\Frequentielle\Freq-Maillage\data/armadillo_light.off')

v , f = utils.ReadObj(path)
print(f'v shape : {v.shape}, f shape {f.shape}')
color = laplacian.diffusion_maps(v,f)

L = laplacian.graph_laplacian(v,f)
v = L @ v

display.display(v,f,color=color)


end = time.time()
global_time = end-start
print(f'time {global_time//60} min {global_time%60} sec')






