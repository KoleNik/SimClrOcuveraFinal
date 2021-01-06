import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

def kinectV2_camera_constants():
    return (365.6,367.195,256.0,212.0)

def projective_to_real(pinhole_image, camera_constants):
    fx,fy,cx,cy = camera_constants

    width = pinhole_image.shape[-1]
    height = pinhole_image.shape[-2]

    p_x,p_y = np.mgrid[0:height:1,0:width:1]
    
    #Compute the real coordinates individually
    z = pinhole_image
    r_y = -z*((p_x - cy)/fy)
    r_x = z*((p_y - cx)/fx)

    #Concatenate them
    z = z.reshape((height,width,1))
    r_y = r_y.reshape((height,width,1))
    r_x = r_x.reshape((height,width,1))

    r = np.concatenate((r_x,r_y,z),axis=2)
    
    return r

def real_to_projective(real_image, camera_constants):
    fx,fy,cx,cy = camera_constants

    r_x = real_image[:,:,0]
    r_y = real_image[:,:,1]
    z = real_image[:,:,2]
    
    p_x = fx*r_x/z + cx
    p_y = -fy*r_y/z + cy
    
    return np.stack([p_x,p_y,z],axis=2)

def d2r(degrees):
    return 3.14159268 * degrees / 180

def pan(angle_in_radians):
    fixed_tilt = d2r(30)
    
    a = angle_in_radians
    #Note some experimentation suggests that the transformations are applied right to left if from_euler
    r = R.from_euler('xyx', [fixed_tilt,angle_in_radians,-fixed_tilt]) 

    return r

def tilt(angle_in_radians):
    return R.from_euler('x',[angle_in_radians])

def roll(angle_in_radians):
    return R.from_euler('z',[angle_in_radians])
    
def rotated_ones(shape,rotation,camera_constants):
    ones = np.ones(shape)
    length = shape[0]*shape[1]
    return rotation.apply(
        projective_to_real(
            ones,camera_constants).reshape(length,3)).reshape((shape[0],shape[1],3))

#The rotation describes what happens to the camera
def rotate_perspective(projective_image, rotation, camera_constants):
    p = projective_image
    
    #Figure out where each point is coming from.
    r1s = rotated_ones(projective_image.shape, rotation, camera_constants)
 
    #Reproject
    h,w = projective_image.shape
    reprojected = real_to_projective(r1s, camera_constants)
    
    #Remap
    map1 = reprojected[:,:,0].reshape((h,w)).astype('float32')
    map2 = reprojected[:,:,1].reshape((h,w)).astype('float32')
    unscaled = cv2.remap(p,map1,map2,cv2.cv2.INTER_LINEAR)

    #Scale
    return unscaled/r1s[:,:,2]