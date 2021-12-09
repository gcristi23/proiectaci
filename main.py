from ctypes import WINFUNCTYPE
from PIL.Image import new
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.data import page,coins
from skimage.color import rgb2gray
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola, threshold_minimum)

import numpy as np

matplotlib.rcParams['font.size'] = 9


image = np.asarray(Image.open("./data/girl2.jpg"))[:,:,:3]
print(image.shape)
gray_image = rgb2gray(image)

print(gray_image.shape)

binary_global = gray_image > threshold_otsu(gray_image)

thresh_min = threshold_minimum(gray_image)
binary_min = gray_image > thresh_min

window_size = 25
thresh_niblack = threshold_niblack(gray_image, window_size=window_size, k=0.8)
thresh_sauvola = threshold_sauvola(gray_image, window_size=window_size)

binary_niblack = gray_image > thresh_niblack
binary_sauvola = gray_image > thresh_sauvola

binaries_list = [binary_global, binary_min, binary_niblack, binary_sauvola]
stacked_binaries = np.stack(binaries_list, axis=2)

new_binary = np.zeros(binary_niblack.shape)
n, m, k = stacked_binaries.shape
th_window_size = 5
current_pixel_weight = 0.7

for i in range(n):
    for j in range(m):
        #daca alb e majoritar o sa fie > 1/2
        current_pixel_th = np.sum(stacked_binaries[i, j, :])/k 

        left = i-th_window_size if i >= th_window_size else 0
        right = i+th_window_size if i < n-th_window_size else n
        top = j-th_window_size if j >= th_window_size else 0
        bottom = j+th_window_size if j < m-th_window_size else m
        #incercam sa vedem ce este majoritar intr-o fereastra
        #daca alb e majoritar o sa fie > 1/2 
        window_th = np.sum(stacked_binaries[right:left, top:bottom, :])/(k*(th_window_size**2))

        #facem o combinatie liniara intre cele 2 majoritati pentru ca vrem sa la ponderam
        #dam mai multa importanta majoritatii de pe pixelul curent
        new_binary[i, j] = 1 if current_pixel_weight*current_pixel_th+(1-current_pixel_weight)*window_th >= 0.5 else 0

plt.figure(figsize=(8, 7))
plt.subplot(2, 3, 1)
plt.imshow(binary_min, cmap=plt.cm.gray)
plt.title('Bimodal Threshold')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Global Threshold')
plt.imshow(binary_global, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(binary_niblack, cmap=plt.cm.gray)
plt.title('Niblack Threshold')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(binary_sauvola, cmap=plt.cm.gray)
plt.title('Sauvola Threshold')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(gray_image, cmap=plt.cm.gray)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(new_binary, cmap=plt.cm.gray)
plt.title('Voting Threshold')
plt.axis('off')


plt.show()