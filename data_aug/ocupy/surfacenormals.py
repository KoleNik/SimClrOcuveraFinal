import numpy as np
import PIL.Image
from .camerageometry import kinectV2_camera_constants, projective_to_real

def normalize_except_where_zero(a):
    a_norm = np.linalg.norm(a, ord=2, axis=2, keepdims=True)
    where_zero = a_norm == 0
    a_norm = np.where(where_zero,1,a_norm)
    return a/a_norm, where_zero

def real_to_surface_normal(r):
    width = r.shape[1]
    height = r.shape[0]

    d_x = np.diff(r, n=1, axis=1)[0:height-1,0:width-1,:]
    d_y = np.diff(r, n=1, axis=0)[0:height-1,0:width-1,:]

    cross_product = -np.cross(d_x,d_y) #I assume this is right, but I'm not sure

    r_norm, r_zero = normalize_except_where_zero(r)
    cross_norm, cross_zero = normalize_except_where_zero(cross_product)

    dot_product = np.sum(r_norm[0:height-1,0:width-1,:]*cross_norm, axis=2)
    
    is_zero = np.logical_or(r_zero[0:height-1,0:width-1,:],cross_zero).reshape((height-1,width-1))
    zeros = np.zeros(dot_product.shape)
    dot_product = np.where(is_zero, zeros, dot_product)
    
    return dot_product

def projective_to_surface_normal(projective, camera_constants = None):
    if camera_constants is None:
        camera_constants = kinectV2_camera_constants()
    return real_to_surface_normal(projective_to_real(projective, camera_constants))

def projective_to_PIL_surface_normal(projective,camera_constants = None):
    if camera_constants is None:
        camera_constants = kinectV2_camera_constants()
    zero_to_one_scale = projective_to_surface_normal(projective, camera_constants)
    return PIL.Image.fromarray((256*np.maximum(0,zero_to_one_scale)).astype("uint8"))

def loadPIL(path, camera_constants = None):
    projective = np.load(path)
    
    return projective_to_PIL_surface_normal(projective, camera_constants)
