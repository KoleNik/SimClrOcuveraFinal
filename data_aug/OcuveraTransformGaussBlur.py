import numpy as np
import cv2 as cv

class OcuveraTransformGauss(object):

    def __init__(self, kernel, borderType):
        self.kernel = kernel
        self.borderType = borderType

    def __call__(self, arr):
        return cv.GaussianBlur(arr, self.kernel, self.borderType)
