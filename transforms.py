import os
import pandas as pd
import matplotlib.image as mpimg
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, image):
        return cv2.resize(image, (self.output_size, self.output_size))
    
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image):
        image_copy = image.data.clone()
        channels = image_copy.shape[0]
        for i in range(channels):
            norm_image = (image_copy[i, :, :] - self.mean[i]) / self.std[i]
            image_copy[i, :, :] = norm_image
        return image_copy.type(torch.FloatTensor)
    

class ToTensor(object):
    def __call__(self, image):
        if(len(image.shape) == 2):
            image = image.reshape(image.shape[0], image.shape[1], 1)
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)   
    
    
class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, image):
        return