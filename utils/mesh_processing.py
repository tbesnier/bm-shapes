import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import os
import open3d as o3d
import copy
import trimesh as tri
import torch
import sys
sys.path.insert(0,"../")
from utils import lddmm_utils

def decimate_mesh(V,F,target):    
    """
    Decimates mesh given by V,F to have number of faces approximatelyu equal to target 
    """
    mesh=getMeshFromData([V,F])
    mesh=mesh.simplify_quadric_decimation(target)
    VS = np.asarray(mesh.vertices, dtype=np.float64) #get vertices of the mesh as a numpy array
    FS = np.asarray(mesh.triangles, np.int64) #get faces of the mesh as a numpy array
    return VS, FS    
    
def subdivide_mesh(V,F,Rho=None,order=1):
    """
    Performs midpoint subdivision. Order determines the number of iterations
    """
    mesh=getMeshFromData([V,F],Rho=Rho)
    mesh = mesh.subdivide_midpoint(number_of_iterations=order)
    VS = np.asarray(mesh.vertices, dtype=np.float64) #get vertices of the mesh as a numpy array
    FS = np.asarray(mesh.triangles, np.int64) #get faces of the mesh as a numpy array
    if Rho is not None:
        RhoS = np.asarray(mesh.vertex_colors,np.float64)[:,0]
        return VS, FS, RhoS
   
    return VS, FS  

def getDataFromMesh(mesh):    
    """
    Get vertex and face connectivity of a mesh
    """
    V = np.asarray(mesh.vertices, dtype=np.float64) #get vertices of the mesh as a numpy array
    F = np.asarray(mesh.triangles, np.int64) #get faces of the mesh as a numpy array  
    color=np.zeros((int(np.size(V)/3),0))
    #if mesh.has_vertex_colors():
    #    color=np.asarray(255*np.asarray(mesh.vertex_colors,dtype=np.float64), dtype=np.int)
    return V, F, color
    
def getMeshFromData(mesh,Rho=None, color=None):    
    """
    Convert Data into mesh object
    """
    V=mesh[0]
    F=mesh[1] 

    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),o3d.utility.Vector3iVector(F))
    
    if Rho is not None:
        Rho=np.squeeze(Rho)
        col=np.stack((Rho,Rho,Rho))
        mesh.vertex_colors =  o3d.utility.Vector3dVector(col.T)
        
    if color is not None:
        mesh.vertex_colors =  o3d.utility.Vector3dVector(color)   
    return mesh

def export_mesh(V, F, file_name):
    """
    Export mesh as .ply file from vertices coordinates and face connectivity
    """
    result = tri.exchange.ply.export_ply(tri.Trimesh(V,F), encoding='ascii')
    output_file = open(file_name, "wb+")
    output_file.write(result)
    output_file.close()
    
    
def varifold_dist(M1, M2):
    """
    Compute varifold distance btw mesh at path M1 and mesh at path M2
    """
    
    # torch type and device
    use_cuda = torch.cuda.is_available()
    torchdeviceId = torch.device("cuda:0") if use_cuda else "cpu"
    torchdtype = torch.float32

    # PyKeOps counterpart
    KeOpsdeviceId = torchdeviceId.index  # id of Gpu device (in case Gpu is  used)
    KeOpsdtype = torchdtype.__str__().split(".")[1]  # 'float32'
    
    def get_data(file):
        if type(file)==str:
            mesh = o3d.io.read_triangle_mesh(file)
        else:
            mesh = file
        V, F, Rho = getDataFromMesh(mesh)
        return(V,F,Rho)
    
    V1, F1, Rho1 = get_data(M1)
    V2, F2, Rho2 = get_data(M2)
    print(F2)
    
    q0 = torch.from_numpy(V1).clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    VT = torch.from_numpy(V2).clone().detach().to(dtype=torchdtype, device=torchdeviceId)
    FS = torch.from_numpy(F1).clone().detach().to(dtype=torch.long, device=torchdeviceId)
    FT = torch.from_numpy(F2).clone().detach().to(dtype=torch.long, device=torchdeviceId)
    sigma = torch.tensor([10], dtype=torchdtype, device=torchdeviceId)
    
    p0 = torch.zeros(q0.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)
    dataloss = lddmm_utils.lossVarifoldSurf(FS, VT, FT, lddmm_utils.GaussLinKernel(sigma=sigma))
    Kv = lddmm_utils.GaussKernel(sigma=sigma)
    loss = lddmm_utils.LDDMMloss(Kv, dataloss)

    dist = loss(p0, q0)
    return dist
    