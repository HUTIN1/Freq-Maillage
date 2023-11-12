import os
import numpy as np
import pywavefront as pwf
import torch

def ReadObj(path):
    path = os.path.abspath(path)
    obj = pwf.Wavefront(path,collect_faces=True)
    # vertices = np.array(obj.vertices)
    vertices = torch.tensor(obj.vertices).cuda()
    faces = [face for mesh in obj.mesh_list for face in mesh.faces]
    # faces = np.array(faces)
    faces = torch.tensor(faces).cuda()
    return vertices, faces
    

def inv_pow(M,pow):
    Dsqrt = M ** (-pow)
    #pass all the inf value to 0
    x, y = torch.where(torch.isinf(Dsqrt))
    Dsqrt[x,y] = 0
    Dsqrt = Dsqrt.type(dtype = torch.float32)
    return Dsqrt