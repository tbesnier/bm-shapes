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
import imageio
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt

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

def create_gif(list_data, file_name):
    filenames = []
    listpq = list_data
    steps = []
    #setting up the faces for the mesh
    rows_unique = np.unique(listpq[0,:,:].T, axis=0, return_index=True)
    new = listpq[0,:,:].T[rows_unique[1]]
    hull = tri.convex.convex_hull(new, qhull_options='QbB Pp Qt QJ')
    faces = hull.faces
    for i in range(listpq.shape[0]):
        steps.append(listpq[i,:,:].T[rows_unique[1]])
    steps = np.array(steps)
    
    for i in range(len(listpq)):

        def frustum(left, right, bottom, top, znear, zfar):
            M = np.zeros((4, 4), dtype=np.float32)
            M[0, 0] = +2.0 * znear / (right - left)
            M[1, 1] = +2.0 * znear / (top - bottom)
            M[2, 2] = -(zfar + znear) / (zfar - znear)
            M[0, 2] = (right + left) / (right - left)
            M[2, 1] = (top + bottom) / (top - bottom)
            M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
            M[3, 2] = -1.0
            return M
        def perspective(fovy, aspect, znear, zfar):
            h = np.tan(0.5*np.radians(fovy)) * znear
            w = h * aspect
            return frustum(-w, w, -h, h, znear, zfar)
        def translate(x, y, z):
            return np.array([[1, 0, 0, x], [0, 1, 0, y],
                         [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)
        def xrotate(theta):
            t = np.pi * theta / 180
            c, s = np.cos(t), np.sin(t)
            return np.array([[1, 0,  0, 0], [0, c, -s, 0],
                         [0, s,  c, 0], [0, 0,  0, 1]], dtype=float)
        def yrotate(theta):
            t = np.pi * theta / 180
            c, s = np.cos(t), np.sin(t)
            return  np.array([[ c, 0, s, 0], [ 0, 1, 0, 0],
                          [-s, 0, c, 0], [ 0, 0, 0, 1]], dtype=float)

        def zrotate(theta):
            t = np.pi * theta / 180
            c, s = np.cos(t), np.sin(t)
            return  np.array([[ c, -s, 0, 0], [ s, c, 0, 0],
                          [0, 0, 1, 0], [ 0, 0, 0, 1]], dtype=float)

        V = np.array(steps[i])
        #V = np.array([it[:,0], it[:,1], it[:,2]]).T
        F = np.array([faces[:,0], faces[:,1], faces[:,2]]).T

        V = (V-(V.max(0)+V.min(0))/2) / max(V.max(0)-V.min(0))
        MVP = perspective(25,1,1,100) @ translate(0,0,-3.5) @ xrotate(120) @ yrotate(180) @ zrotate(-20)
        V = np.c_[V, np.ones(len(V))]  @ MVP.T
        V /= V[:,3].reshape(-1,1)
        V = V[F]
        T =  V[:,:,:2]
        Z = -V[:,:,2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z-zmin)/(zmax-zmin)
        C = plt.get_cmap("magma")(Z)
        I = np.argsort(Z)
        T, C = T[I,:], C[I,:]
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1], aspect=1, frameon=False)
        collection = PolyCollection(T, closed=True, linewidth=0.1, facecolor=C, edgecolor="black")
        ax.add_collection(collection)
    
        # create file name and append it to a list
        filename = f'{i}.png'
        filenames.append(filename)
    
        # last frame of each viz stays longer
        if (i == len(listpq)-1):
                for i in range(5):
                    filenames.append(filename)
    
        # save frame
        plt.savefig(filename)
        plt.close()# build gif
    
    with imageio.get_writer(file_name, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    