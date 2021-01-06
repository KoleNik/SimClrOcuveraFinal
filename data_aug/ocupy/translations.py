import numpy as np
import math

def kinectV2_camera_constants():
    return (365.6,367.195, 212.0,256.0)

def translation(pinhole_image, offset):
    fx, fy, cx, cy = kinectV2_camera_constants()
    v1, v2, v3 = offset
    new = np.zeros((pinhole_image.shape[0], pinhole_image.shape[1]))

    for i in range(new.shape[0]):
        for j in range(new.shape[1]):
            if pinhole_image[i, j] > 0:
                px = math.floor((-fx / pinhole_image[i, j]) * (pinhole_image[i, j] * (i - cx) / (-fx) - v1) + cx)
                py = math.floor((-fy / pinhole_image[i, j]) * (pinhole_image[i, j] * (j - cy) / (-fy) - v2) + cy)
                if 0 <= px and px <= 423 and 0 <= py and py <= 511:
                    if new[px, py] > 0:
                        new[px, py] = min(new[px, py], pinhole_image[i, j])
                    else:
                        new[px, py] = pinhole_image[i, j]
    return new

