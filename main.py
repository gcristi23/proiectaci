import matplotlib.pyplot as plt
from utils import get_numpy_data, get_stacked_binaries, get_output, conf, compute_threshold, plot
import numpy as np


thresh_vote = 0.38
current_pixel_weight = 0.5



if __name__ == '__main__':
    label_input = "input"
    label_output = "output"
    
    if conf["test"]:
         label_input += "_test"
         label_output += "_test"
    input_path = conf["input_path"] if not conf["test"] else conf["test_path"]
    stacked_binaries, thresh_list = get_numpy_data(input_path, get_stacked_binaries, label_input)
    if not conf["test"]:
        get_numpy_data(conf["output_path"], get_output, label_output)
   
    if conf["test"]:
        stacked_binaries = stacked_binaries[0]
        thresh = thresh_list[0]

        binary_sauvola = stacked_binaries[3]
        binary_niblack = stacked_binaries[2]
        binary_global = stacked_binaries[0]
        gray_image = stacked_binaries[4]
        binary_min = stacked_binaries[1]
        thresh_sauvola = thresh[2]
        thresh_global = thresh[0]
        
        stacked_binaries = stacked_binaries.T
        majority_binary = compute_threshold(stacked_binaries, 0, 0) > 0.5
        majority_binary = majority_binary.T
        print(majority_binary.shape)
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