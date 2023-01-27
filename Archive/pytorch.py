
import numpy as np
np.random.seed(1773)

import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch_model import Net

from sklearn.model_selection import train_test_split


import operator
# %matplotlib inline

plt.rcParams['figure.figsize'] = (8.0, 8.0)
from math import sqrt



def plotExample(row, variances, xvalues, variancePredictions = None):
    # draw a single case
    
    xrange = np.linspace(-4,+4,101)
    
    from scipy.stats import norm
    
    #----------
    # plot the Gaussian
    #----------

    pdf = norm(loc = 0, scale = sqrt(variances[row])).pdf
    
    yvalues = pdf(xrange)
    plt.plot(xrange, yvalues)

    #----------

    ymin = min(yvalues); ymax = max(yvalues)
    
    ytrue  = 0.6 * ymax + 0.4 * ymin
    yestU  = 0.5 * ymax + 0.5 * ymin
    yestML = 0.4 * ymax + 0.6 * ymin
    ypred  = 0.3 * ymax + 0.7 * ymin
    
    #----------
    # plot points on Gaussian
    #----------

    yvalues = pdf(xvalues[row])
    plt.plot(xvalues[row], yvalues, 'o')
    
    #----------
    # plot true sigma
    #----------
    sigma = sqrt(variances[row])
    plt.plot([ - sigma, + sigma ], [ytrue, ytrue], label = 'true (%.2f)' % sigma)
    
    #----------
    # plot ML and unbiased estimator of the variance
    #----------
    
    for label, ddof, ypos in (
       ('unbiased est.', 1, yestU),
       ('ML est.', 0, yestML),
    ):
    
        sigma = sqrt(np.var(xvalues[row], ddof = ddof))
        plt.plot([ - sigma, + sigma ], [ypos, ypos], label = label + ' (%.2f)' % sigma)
    

    #----------
    # plot prediction
    #----------
    if variancePredictions is not None:
        sigma = sqrt(variancePredictions[row])
        
        plt.plot([ - sigma, + sigma ], [ypred, ypred], label = 'predicted (%.2f)' % sigma)

    
    
    plt.grid()

    plt.legend()


# range of variances to generate
varianceRange = [ 0.5, 1.5 ]

# total number of points to generate
numRows = 10000

# minimum and maximum number of points to draw from each distribution 
# (both inclusive)
numPointsRange = [ 10, 20 ]


# generate true variances of Gaussians
# convert to float32 to avoid incompatible data types during training
trueVariances = np.random.uniform(
                   varianceRange[0], varianceRange[1], numRows).astype('float32')

trueSigmas = np.sqrt(trueVariances)

# determine how many points should be drawn from each Gaussian
numPoints = np.random.randint(numPointsRange[0], numPointsRange[1] + 1, size = numRows)

# draw a set of points from the Gaussian
xvalues = []

for row in range(numRows):
    thisNumPoints = numPoints[row]

    # draw points from this Gaussian
    xv = np.random.normal(loc = 0, scale = trueSigmas[row], size = (thisNumPoints,1))
    
    # convert to float32 to avoid problems with incompatible data types during training
    xvalues.append(xv.astype('float32'))
    



# calculate ML estimators for each point
mlEstimators = np.array([ np.var(xv, ddof = 0) for xv in xvalues], dtype = 'float32')

# calculate unbiased estimators for each point
ubEstimators = np.array([ np.var(xv, ddof = 1) for xv in xvalues], dtype = 'float32')

(inputTrain, inputTest, 
 targetTrain, targetTest,
 indicesTrain, indicesTest) = train_test_split(xvalues, 
                                               trueVariances, 
                                               # mlEstimators,
                                               # trueMeans,
                                               np.arange(numRows),
                                               test_size=0.20, random_state=42)


print(inputTrain)

# instantiate the model
model = Net()

allIndices = np.arange(len(targetTrain))

# define the loss function
lossFunc = nn.MSELoss()

minibatchSize = 32

# number of training epochs
numEpochs = 5 #40

optimizer = optim.Adam(model.parameters(), lr = 0.0001)

trainLosses = []; testLosses = []

# variable for target values of test set
testTargetVar = Variable(torch.from_numpy(np.stack(targetTest)))

print("starting training")
for epoch in range(numEpochs):

    np.random.shuffle(allIndices)
    
    # put model in training mode
    model.train()
    
    trainLoss  = 0
    trainSteps = 0
    
    for indices in np.array_split(allIndices, minibatchSize):
        
        optimizer.zero_grad()
    
        # forward through the network
        output = model.forward([ inputTrain[index] for index in indices])
    
        # build a PyTorch variable with the target value
        # so that we can propagate backwards afterwards
        thisTarget = Variable(
            torch.from_numpy(np.stack([ targetTrain[index] for index in indices ])))
    
        # calculate loss
        loss = lossFunc.forward(output, thisTarget)
   
        # accumulate 
        # trainLoss += loss.data[0]
        trainLoss += loss.data
        
        print("test1")
        # backpropagate 
        loss.backward()

        print("test1a")
                            
        # update learning rate        
        optimizer.step()

        print("testba")

        
        trainSteps += 1

        print("test2")

        
    trainLoss /= trainSteps
    trainLosses.append(trainLoss)    

    # evaluate model on test set
    model.eval()
        
    output = model.forward(inputTest)
            
    # calculate loss on test set
    testLoss = lossFunc.forward(output, testTargetVar).data  #[0]

    testLosses.append(testLoss)
    
    print("epoch",epoch,"train loss=", trainLoss, "test loss=",testLoss)








# instantiate the model
model = Net()

allIndices = np.arange(len(targetTrain))

# define the loss function
lossFunc = nn.MSELoss()

minibatchSize = 32

# number of training epochs
numEpochs = 40

optimizer = optim.Adam(model.parameters(), lr = 0.0001)

trainLosses = []; testLosses = []

# variable for target values of test set
testTargetVar = Variable(torch.from_numpy(np.stack(targetTest)))

print("starting training")
for epoch in range(numEpochs):

    np.random.shuffle(allIndices)
    
    # put model in training mode
    model.train()

    
    trainLoss  = 0
    trainSteps = 0
    
    for indices in np.array_split(allIndices, minibatchSize):
        
        optimizer.zero_grad()
    
        # forward through the network
        output = model.forward([ inputTrain[index] for index in indices])
    
        # build a PyTorch variable with the target value
        # so that we can propagate backwards afterwards
        thisTarget = Variable(
            torch.from_numpy(np.stack([ targetTrain[index] for index in indices ])))
    
        # calculate loss
        loss = lossFunc.forward(output, thisTarget)
   
        # accumulate 
        trainLoss += loss.data  #[0]

        # backpropagate 
        loss.backward()
                            
        # update learning rate        
        optimizer.step()
        
        trainSteps += 1
        
    trainLoss /= trainSteps
    trainLosses.append(trainLoss)    
        
    # evaluate model on test set
    model.eval()
        
    output = model.forward(inputTest)
            
    # calculate loss on test set
    testLoss = lossFunc.forward(output, testTargetVar).data  #[0]

    testLosses.append(testLoss)
    
    print("epoch",epoch,"train loss=", trainLoss, "test loss=",testLoss)
