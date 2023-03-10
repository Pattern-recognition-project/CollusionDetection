import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import operator

import numpy as np
import random
import os
import matplotlib.pyplot as plt
import math

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

class Net(nn.Module):

    def __init__(self,
                 numLayersInputSide  = 3,
                 widthInputSide      = 16,
                 numLayersOutputSide = 4,
                 widthOutputSide     = [128,64,32],
                ):

        super(Net, self).__init__()

        #----------
        # input side layers
        #----------

        # NETWORK 1
        numInputs = 1

        self.inputSideLayers = []

        for i in range(numLayersInputSide):
            layer = nn.Linear(numInputs, widthInputSide, dtype=torch.float64)
            self.inputSideLayers.append(layer)
            self.add_module("iLayer%d" % i, layer)

            numInputs = widthInputSide

        #----------
        # output side layers
        #----------

        # NETWORK 2
        numNewfeatures = 24 # to add the new features as another input to the second model
        numInputs = [widthInputSide + numNewfeatures] + widthOutputSide

        self.outputSideLayers = []
        for i in range(numLayersOutputSide):

            if i == numLayersOutputSide - 1:
                layer = nn.Linear(widthOutputSide[-1], 1, dtype=torch.float64)
                self.outputSideLayers.append(layer)
                self.add_module("oLayer%d" % i, layer)
            else:
                layer = nn.Linear(numInputs[i], widthOutputSide[i], dtype=torch.float64)
                self.outputSideLayers.append(layer)
                self.add_module("oLayer%d" % i, layer)

        self.dropout = nn.Dropout(0.25)
    #----------------------------------------

    def forward(self, points, added_features): # added_features are the variables like skewness, kurtosis, etc
        # points is one batch, so a collection of auctions: each auction is an array of arrays (the internal ones are the bids)


        # overall output for the entire minibatch
        outputs = []

        # loop over minibatch entries
        for indexAuction, auction in enumerate(points):         # loop on the auctions

            # auction is a list of 1D tensors (ogni bids ?? a sua volta un array)
            # stack all input points into a 2D tensor
            # (the second dimension will have size 1)
            h = np.stack(auction)

            h = Variable(torch.from_numpy(h)) # A PyTorch Variable is a wrapper around a PyTorch Tensor,
                                              # and represents a node in a computational graph. If x is a Variable
                                              # then x.data is a Tensor giving its value, and x.grad is another Variable
                                              # holding the gradient of x with respect to some scalar value.

            # forward all input points through the input side network
            # each auction passes through each layer and relu
            # we are passing a vector (so a signel input) containing all the bids for each auction
            for layer in self.inputSideLayers:
                h = layer(h)
                h = torch.relu(h)

            # average the input side network outputs: sum along first dimension (point index),
            # then divide by number of points
            # (it aggregates the output tensors of the first network).

            output = h.sum(0) / len(auction)

            # forward step of the second network

            # add dropout layer
            h = self.dropout(output)


            # passing through the second network
            for layerIndex, layer in enumerate(self.outputSideLayers):
                if layerIndex==0: # if we are in the first layer we have to consider the new features
                    new_features = Variable(torch.from_numpy(np.asarray(added_features[indexAuction])))
                    h = torch.cat((h,new_features))
                    h = layer(h)
                else:
                    h = layer(h)

                # h= layer(h)

                if layerIndex != len(self.outputSideLayers) - 1:
                    h = torch.relu(h)
                else:
                    h = torch.sigmoid(h) # to have values between 0 and 1 (needed if the chosen loss function is BCEloss)

            # contains the output for each auction in the present batch, so at the end should be of the same length of points
            outputs.append(h)

        # end of loop over minibatch entries

        # convert the list of outputs to a torch 2D tensor

        return torch.cat(outputs, 0)
