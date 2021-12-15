import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola, threshold_minimum)

import numpy as np

matplotlib.rcParams['font.size'] = 9
thresh_vote = 0.38
th_window_size = 1
current_pixel_weight = 0.5

def plot(image, title, row, col, pos):
    plt.subplot(row, col, pos)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(title)
    plt.axis('off')


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

image = np.asarray(Image.open("./data/girl.png"))[:,:,:3]
gray_image = rgb2gray(image)

thresh_global = threshold_otsu(gray_image)
binary_global = gray_image > thresh_global

thresh_min = threshold_minimum(gray_image)
binary_min = gray_image > thresh_min

window_size = 25
thresh_niblack = threshold_niblack(gray_image, window_size=window_size, k=0.5)
thresh_sauvola = threshold_sauvola(gray_image, window_size=window_size)

binary_niblack = gray_image > thresh_niblack
binary_sauvola = gray_image > thresh_sauvola

binaries_list = [binary_global, binary_min, binary_niblack, binary_sauvola]
stacked_binaries = np.stack(binaries_list, axis=2)

global_binaries = np.stack([binary_global, binary_min], axis=2)
local_binaries = np.stack([binary_niblack, binary_sauvola], axis=2)

majority_binary = compute_threshold(stacked_binaries, 0, 0) > 0.5

n, m = binary_sauvola.shape

global_threshold = compute_threshold(binary_global.reshape(n,m,1), 0)
local_threshold = compute_threshold(binary_niblack.reshape(n,m,1), 1)

final = (global_threshold < local_threshold) * binary_niblack
final += (global_threshold > local_threshold) * binary_global

global_distance_to_original = np.abs(thresh_global - gray_image)
local_ditance_to_original = np.abs(thresh_sauvola - gray_image)

final_distance = (global_distance_to_original > local_ditance_to_original) * binary_sauvola
final_distance += (global_distance_to_original <= local_ditance_to_original) * binary_global
 
to_plot = [
    (binary_min, "Bimodal Threshold"),
    (binary_global, "Global Threshold"),
    (binary_niblack, "Niblack Threshold"),
    (binary_sauvola, "Sauvola Threshold"),
    (gray_image, "Original"),
    (final, "Voting Result"),
    (global_threshold, "Global "),
    (local_threshold, "Local "),
    (majority_binary, "Majority Voting"),
    (global_distance_to_original, "Global Distance"),
    (local_ditance_to_original, "Local Distance"),  
    (final_distance, "Final Distance Result"),     
]
plt.figure(figsize=(8, 7))
row = 4
col = int(np.ceil(len(to_plot)/row))

for i, p in enumerate(to_plot):
    plot(p[0], p[1], row, col, i+1)

plt.show()