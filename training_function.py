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


def binary_output(output):
    # actual binary predictions
    out = output.detach().numpy()
    return np.where(out>=0.5,1,0)

def training_function(numberBatches, numEpochs, inputTrain, targetTrain, inputTest, targetTest, added_featuresTrain, added_featuresTest, lr=0.01):

    # instantiate the model
    model = Net()

    # define the loss function
    lossFunc = nn.BCELoss() #if this is used the sigmoid layer should be added in the second network
    # lossFunc = nn.BCEWithLogitsLoss()

    allIndices = np.arange(len(targetTrain))

    numberBatches = numberBatches

    # number of training epochs
    numEpochs = numEpochs

    optimizer = optim.Adam(model.parameters(), lr=lr)


    trainLosses_epoch =[]
    testLosses_epoch = []
    
    print("starting training")
    for epoch in range(numEpochs):
        

        np.random.shuffle(allIndices)
        
        # put model in training mode
        model.train()
        
        trainLoss  = 0 
        trainSteps = 0

        for indices in np.array_split(allIndices, numberBatches): # splitta gli indici shuffled in numberBatches vettori, ognuno che mi definisce un batch
            # indices colleziona quali auctions voglio, dentro ho i riferimenti alle varie auction 
            # batchSize = len(indices)
            optimizer.zero_grad()
        
            # forward through the network
            # inpuTrain[index] is each auction (array with all the bids)

            # added features for each auction in this batch should be passed as input
            output = model.forward([ inputTrain[index] for index in indices], [added_featuresTrain[index] for index in indices])

        
            # build a PyTorch variable with the target value
            # so that we can propagate backwards afterwards
            # targets of the auction in the current batch
            thisTarget = Variable(torch.from_numpy(np.stack([ targetTrain[index] for index in indices ])))
            # output = Variable(torch.from_numpy(binary_output(output).astype(float)))
            # output.requires_grad_()
            # calculate loss of the present batch 
            loss = lossFunc(output, thisTarget)
            
            # backpropagate (considering the averaged loss on the present batch)
            loss.backward()
                                
            # update learning rate        
            optimizer.step()
            
            # accumulate over batches to get the epoch loss
            trainLoss += loss
            trainSteps += 1

        # saving training losses for each epoch (average of the losses in each batch)
        loss_epoch = trainLoss/trainSteps
        trainLosses_epoch.append(loss_epoch)   
                
        model.eval()
        # evaluate model on test set
        output = model.forward(inputTest, added_featuresTest)
        
        # computing test loss 
        testTargetVar = Variable(torch.from_numpy(np.stack(targetTest)))    
        
        # loss on the test set
        testLoss = lossFunc(output,testTargetVar)

        # saving test losses for each epoch
        testLosses_epoch.append(testLoss)


    

        if epoch % 5==0: #print the last losses added
            print("epoch",epoch,"train loss =", trainLosses_epoch[-1] , "test loss =", testLosses_epoch[-1])  
            #print("output: ", output)
 
    return model, output, trainLosses_epoch, testLosses_epoch