import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import operator

import numpy as np
np.random.seed(1773)

import matplotlib.pyplot as plt
import math





class Net(nn.Module):
    
    def __init__(self, 
                 numLayersInputSide  = 5, 
                 widthInputSide      = 50,
                 numLayersOutputSide = 5,
                 widthOutputSide     = 50,
                ):
        
        super(Net, self).__init__()
        
        #----------
        # input side layers
        #----------

        numInputs = 1
        
        self.inputSideLayers = []
        for i in range(numLayersInputSide):
            layer = nn.Linear(numInputs, widthInputSide)
            self.inputSideLayers.append(layer)
            self.add_module("iLayer%d" % i, layer)
            
            numInputs = widthInputSide

        #----------
        # output side layers
        #----------

        numInputs = widthInputSide
        numOutputs = widthOutputSide
        
        self.outputSideLayers = []
        for i in range(numLayersOutputSide):
          
            if i == numLayersOutputSide - 1:
                # we want to learn the variance
                numOutputs = 1
            else:
                numOutputs = widthOutputSide
            
            layer = nn.Linear(numInputs, numOutputs)
            self.outputSideLayers.append(layer)
            self.add_module("oLayer%d" % i, layer)
            
            numInputs = numOutputs
             
    #----------------------------------------
    
    def forward(self, points):
        
        # points is a list of list if 2D points
        # the first index is the minibatch index, the second index is the index
        # of the point within the row
                    
        # overall output for the entire minibatch
        outputs = []
        
        # loop over minibatch entries
        for thisPoints in points:

            # outputs of each point of this minibatch entry
            thisOutputs = [ ]
            
            # thisPoints is a list of 1D tensors
            # stack all input points into a 2D tensor
            # (the second dimension will have size 1)
            h = np.stack(thisPoints)
            
            h = Variable(torch.from_numpy(h))
            
            # forward all input points through the input side network
            for layer in self.inputSideLayers:
                h = layer(h)
                h = F.relu(h)
                            
            # average the input side network outputs: sum along first dimension (point index), 
            # then divide by number of points
            output = h.sum(0) / len(thisPoints)
            
            # feed through the output side network
            h = output
            for layerIndex, layer in enumerate(self.outputSideLayers):
                
                h = layer(h)

                # note: since we want to do regression, we do NOT 
                # apply a nonlinearity after the last layer
                
                if layerIndex != len(self.outputSideLayers) - 1:
                    h = F.relu(h)
                
            outputs.append(h)
            
        # end of loop over minibatch entries
         
        # convert the list of outputs to a torch 2D tensor
        return torch.cat(outputs, 0)            
