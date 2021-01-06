import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2 as cv

from .ocupy import surfacenormals
from PIL import Image

class OcuveraDataSet(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imagePath = os.listdir(root_dir)

    def __len__(self):
        return len(self.imagePath)

    def __getitem__(self, idx):
        arr = np.load(self.root_dir + "/" + self.imagePath[idx])
        arr = cv.GaussianBlur(arr,(5,5),0)


        #arr = np.vstack((arr, arr, arr))
        arr = arr.reshape((212, 256, 1))
        arr = np.concatenate((arr, arr, arr), axis = 2)
        arr = arr.astype(np.float32)
        #img = surfacenormals.projective_to_PIL_surface_normal(arr)
        #img = Image.fromarray(arr)
        result = self.transform(arr)
        return result, 1

    def transforms(arr):
        arr = cv.GaussianBlur(arr,(5,5),0)

        return arr
