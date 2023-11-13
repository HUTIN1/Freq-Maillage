import os
import numpy as np
import pywavefront as pwf
import torch


class OFFReader():
    def __init__(self):
        FileName = None
        Output = None

    def SetFileName(self, fileName):
        self.FileName = fileName

    def GetOutput(self):
        return self.Output

    def Update(self):
        with open(self.FileName) as file:

            first_string = file.readline() # either 'OFF' or 'OFFxxxx xxxx x'

            if 'OFF' != first_string[0:3]:
                raise('Not a valid OFF header!')

            elif first_string[3:4] != '\n':
                new_first = 'OFF'
                new_second = first_string[3:]
                n_verts, n_faces, n_dontknow = tuple([int(s) for s in new_second.strip().split(' ')])		

            else:
                n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])

            v = []
            f = []
            for i_vert in range(n_verts):
                p = [float(s) for s in file.readline().strip().split(' ')]
                v.append([p[0], p[1], p[2]])

            for i_face in range(n_faces):
                
                t = [int(s) for s in file.readline().strip().split(' ')]
                f.append([t[1],t[2],t[3]])

        self.Output= {'vertices':v,'faces':f}
    
def ReadObj(path):
    path = os.path.abspath(path)
    _ , extension = os.path.splitext(path)
    extension = extension.lower()
    if extension == ".obj":
        obj = pwf.Wavefront(path,collect_faces=True)
        # vertices = np.array(obj.vertices)
        vertices = torch.tensor(obj.vertices).cuda()
        faces = [face for mesh in obj.mesh_list for face in mesh.faces]
        # faces = np.array(faces)
        faces = torch.tensor(faces).cuda()
    elif extension == '.off':
        reader = OFFReader()
        reader.SetFileName(path)
        reader.Update()
        surf = reader.GetOutput()
        vertices = torch.tensor(surf['vertices']).cuda()
        faces = torch.tensor(surf['faces']).cuda()
    return vertices, faces
    

def inv_pow(M,pow):
    Dsqrt = M ** (-pow)
    #pass all the inf value to 0
    x, y = torch.where(torch.isinf(Dsqrt))
    Dsqrt[x,y] = 0
    Dsqrt = Dsqrt.type(dtype = torch.float32)
    return Dsqrt



