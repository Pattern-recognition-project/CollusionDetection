from data import Data
import pickle

import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import random

from torch_model_G import Net

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


def training_function(minibatchSize, numEpochs, inputTrain, targetTrain, inputTest, targetTest, lr=0.001):

    # instantiate the model
    model = Net()

    # define the loss function
    # lossFunc = nn.BCELoss() #if this is used the sigmoid layer should be added in the second network
    lossFunc = nn.BCEWithLogitsLoss()

    allIndices = np.arange(len(targetTrain))

    minibatchSize = minibatchSize

    # number of training epochs
    numEpochs = numEpochs

    # optimizer = optim.SGD(model.parameters(), lr = 0.0001)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainLosses = []
    testLosses = []

    
    print("starting training")
    for epoch in range(numEpochs):
        trainLosses_epoch =[]
        testlosses_epoch = []
        np.random.shuffle(allIndices)
        
        # put model in training mode
        model.train()
        
        trainLoss  = 0
        trainSteps = 0

        for indices in np.array_split(allIndices, minibatchSize):
            
            optimizer.zero_grad()
        
            # forward through the network
            # inpuTrain[index] is each auction
            # added features for each auction in this batch should be passed as input

            output = model.forward([ inputTrain[index] for index in indices]) #, [added_featuresTrain[index] for index in indices])

        
            # build a PyTorch variable with the target value
            # so that we can propagate backwards afterwards
            thisTarget = Variable(torch.from_numpy(np.stack([ targetTrain[index] for index in indices ])))
        
            # calculate loss
            # loss = lossFunc.forward(output, thisTarget)
            loss = lossFunc(output, thisTarget)
            
            # accumulate 
            trainLoss += loss
            
            # backpropagate 
            loss.backward()
                                
            # update learning rate        
            optimizer.step()
            

            trainSteps += 1

        # compute training loss of one auction 
        trainLoss /= trainSteps

        # saving losses of each auction (averaged)
        trainLosses_epoch.append(trainLoss)   
                
        model.eval()
            
        output = model.forward(inputTest)
        
        # computing test loss 
        testTargetVar = Variable(torch.from_numpy(np.stack(targetTest)))    # evaluate model on test set

        testLoss = lossFunc(output,testTargetVar)
        testlosses_epoch.append(testLoss)
    
        finaltrainLoss_epoch = sum(trainLosses_epoch)/len(trainLosses_epoch)
        finaltestLoss_epoch = sum(testlosses_epoch)/len(testlosses_epoch)

        # losses averaged for each epoch
        trainLosses.append(finaltrainLoss_epoch)
        testLosses.append(finaltestLoss_epoch)

        if epoch % 5==0:
            print("epoch",epoch,"train loss =", finaltrainLoss_epoch , "test loss =", finaltestLoss_epoch)  
 
    return model, output, trainLosses, testLosses