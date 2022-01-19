from os import extsep
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDeepVoting(nn.Module):
    def __init__(self):
        super(ConvDeepVoting, self).__init__()
       
        self.conv1 = nn.Conv2d(5, 3, 3, padding=1)  
        self.conv2 = nn.Conv2d(3, 2, 3, padding=1)
        self.conv3 = nn.Conv2d(2, 1, 3, padding=1)


    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = F.sigmoid(self.conv2(x))
        x = F.sigmoid(self.conv3(x))
              
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(5, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
              
        return x


class ConvVoting(nn.Module):
    def __init__(self):
        super(ConvVoting, self).__init__()
       

        self.conv1 = nn.Conv2d(5, 1, 1, padding=0) 


    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
              
        return x


models = {
    "autoencoder": ConvAutoencoder,
    "voting": ConvVoting,
    "deep_voting": ConvDeepVoting
}