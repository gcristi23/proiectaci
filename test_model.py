from os import extsep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from models import models
from utils import get_device, conf, plot

if __name__ == "__main__":

    #Epochs
    n_epochs = 30
    batch_size = 4
    dataset_size = 2500

    input = np.load("input_test0.npy")
    input_image = torch.from_numpy(input).float()

    model = models[conf["model"]]()
    model.load_state_dict(torch.load(conf["model"]))

    device = get_device()

    model.to(device)
    image_tensor = input_image.to(device)

    output = model(image_tensor)
    output_cpu = output.cpu()

    for i in range(output.size(0)):
        image = output_cpu[i].detach().numpy()[0,:,:]
        plt.hist(image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')

        plt.figure(figsize=(9,9))
        
        plot(image, "Output", 2,3,1)
        for j in range(5):
            plot(input[i,j,:,:],"Input"+str(j),2,3,j+2)
        plt.show()