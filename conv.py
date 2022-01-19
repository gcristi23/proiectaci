from os import extsep
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_device, conf
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import json
from models import models


if __name__ == "__main__":

    #Epochs
    n_epochs = conf["n_epochs"]
    batch_size = conf["training_batch_size"]
    dataset_size = conf["maximum_data_size"]

    input = np.load(f"input{conf['batch_no']}.npy")
    output = np.load(f"output{conf['batch_no']}.npy")

    trainset_input = torch.utils.data.TensorDataset(torch.from_numpy(input[:dataset_size,:,:,:]).float())
    train_input_loader = torch.utils.data.DataLoader(trainset_input, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    trainset_output = torch.utils.data.TensorDataset(torch.from_numpy(output[:dataset_size,:,:,:]).float())
    train_output_loader = torch.utils.data.DataLoader(trainset_output, batch_size=batch_size,
                                            shuffle=False, num_workers=1)


    dataiter = iter(train_input_loader)
    images = dataiter.next()
    images = images[0].numpy() # convert images to numpy for display

    # print(images.shape)
    # imshow(images[0,0,:,:])
    # plt.show()

    output_images = iter(train_output_loader).next()[0].numpy()
    print(output_images.shape, images.size)

    model = models[conf["model"]]()
    try:
        model.load_state_dict(torch.load(conf["model"]))
    except:
        print("there is no model saved")

    print(model)

    #Loss function
    criterion = nn.BCELoss()

    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



    device = get_device()
    print(device)
    model.to(device)

    losses = []
    try:
        with open(f"{conf['model']}_train_loss") as f:
            losses = json.load(f)
    except:
        print("no losses saved")
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        output_iter = iter(train_output_loader)
        #Training
        for data in train_input_loader:
            images = data[0]
            images_tensor = images.to(device)

            output_images = output_iter.next()[0]
            output_tensor = output_images.to(device)
            optimizer.zero_grad()
            outputs = model(images_tensor)

            loss = criterion(outputs, output_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images_tensor.size(0)

        train_loss = train_loss/len(train_input_loader)
        losses.append(train_loss)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    with open(f"{conf['model']}_train_loss","w") as f:
        json.dump(losses, f)

    torch.save(model.state_dict(), conf["model"])