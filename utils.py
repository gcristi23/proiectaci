import json
from traceback import print_tb
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola, threshold_minimum, threshold_li)

import matplotlib.pyplot as plt
import matplotlib
import os
import torch


with open("config.json") as f:
    conf = json.load(f)

matplotlib.rcParams['font.size'] = 9
th_window_size = conf["th_window_size"]

def compute_threshold(input_binaries, bias, weight=1):
    n, m, k = input_binaries.shape
    pixel_image = np.zeros(input_binaries.shape[:2])
    window_image = np.zeros(input_binaries.shape[:2])
    for i in range(n):
        for j in range(m):
            #daca alb e majoritar o sa fie > 1/2
            current_pixel_sum = np.sum(input_binaries[i, j, :])
            current_pixel_th = current_pixel_sum/k 

            left = j-th_window_size if j >= th_window_size else 0
            right = j+th_window_size if j < m-th_window_size else n
            top = i-th_window_size if i >= th_window_size else 0
            bottom = i+th_window_size if i < n-th_window_size else m
            right+=1
            bottom+=1
            #incercam sa vedem ce este majoritar intr-o fereastra
            #daca alb e majoritar o sa fie > 1/2 
            window = input_binaries[top:bottom, left:right, :]
            window_sum = np.sum(window)-current_pixel_sum
            n_elements = ((right-left)*(bottom-top)*k)-k
            window_th = window_sum/n_elements

            #facem o combinatie liniara intre cele 2 majoritati pentru ca vrem sa la ponderam
            #dam mai multa importanta majoritatii de pe pixelul curent
            pixel_image[i, j] = np.abs(bias-current_pixel_th)
            window_image[i, j] = window_th
    new_binary = pixel_image-window_image*weight
    
    return np.abs(new_binary)

def get_stacked_binaries(path):
    image = np.asarray(Image.open(path))[:,:,:3]

    gray_image = rgb2gray(image)
    
    thresh_global = threshold_otsu(gray_image)
    binary_global = gray_image > thresh_global

    thresh_min = threshold_li(gray_image)
    binary_min = gray_image > thresh_min

    window_size = 25
    thresh_niblack = threshold_niblack(gray_image, window_size=window_size, k=0.5)
    thresh_sauvola = threshold_sauvola(gray_image, window_size=window_size)

    binary_niblack = gray_image > thresh_niblack
    binary_sauvola = gray_image > thresh_sauvola

    binaries_list = [binary_global, binary_min, binary_niblack, binary_sauvola, gray_image]
    stacked_binaries = np.stack(binaries_list)
    thresh_global = np.ones(thresh_niblack.shape)*thresh_global
    thresh_list = [thresh_global, thresh_niblack, thresh_sauvola]

    stacked_thresh = np.stack(thresh_list)
    return stacked_binaries[:,:,:], stacked_thresh

def get_output(path):
    image = np.asarray(Image.open(path))
    if len(image.shape) == 3:
        return image[:, :, 0].reshape(1,200,-1)
    else:
        return image.reshape(1,200,-1)

def plot(image, title, row, col, pos):
    plt.subplot(row, col, pos)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(title)
    plt.axis('off')


def get_numpy_data(data_path, processing, label):
    images = os.listdir(data_path)
    
    data = []
    thresh_stack = []
    for image_name in images:
        try:
            a=processing(os.path.join(data_path, image_name))

            if isinstance(a, tuple):
                thresh_stack.append(a[1])
                a = a[0]
            if(np.max(a)>1):
                a=a//255
            data.append(a)
        except Exception as e:
            print(image_name)
            print(e)
            exit(0)

    stacked_data = np.stack(data)
    print(f"{label} shape:"+str(stacked_data.shape))

    for i, j in enumerate(range(0,stacked_data.shape[0], conf["batch_size"])):
        data = stacked_data[j:j+conf["batch_size"],:,:,:]
        np.save(f"{label}{i}",data)
    return stacked_data, thresh_stack


def imshow(img):
    img = img / 2 + 0.5  
    plt.figure()
    plt.imshow(img,cmap = "gray")

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device