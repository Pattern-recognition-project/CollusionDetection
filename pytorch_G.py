from data import Data
import pickle

import numpy as np
np.random.seed(1773)

import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from torch_model_G import Net


from math import sqrt


with open("DB_Collusion_All_processed.obj","rb") as filehandler:
    data = pickle.load(filehandler)


# get train and test data and convert in the correct form
inputTrain = data.get_train_X()
targetTrain = data.get_train_y()
inputTest = data.get_test_X()
targetTest = data.get_test_y()
inputTrain = [[[value] for value in auction] for auction in inputTrain]
inputTest = [[[value] for value in auction] for auction in inputTest]


# consider features to be added for the second network
added_features = data.load_aggegrated()
added_featuresTrain = added_features[:len(inputTrain)]

# instantiate the model
model = Net()

allIndices = np.arange(len(targetTrain))

# define the loss function
# lossFunc = nn.BCELoss() #if this is used the sigmoid layer should be added in the second network
lossFunc = nn.BCEWithLogitsLoss()

minibatchSize = 32

# number of training epochs
numEpochs = 10

optimizer = optim.SGD(model.parameters(), lr = 0.0001)

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
        # inpuTrain[index] is each auction
        # added features for each auction in this batch should be passed as input

        output = model.forward([ inputTrain[index] for index in indices]) #, [added_featuresTrain[index] for index in indices])

    
        # build a PyTorch variable with the target value
        # so that we can propagate backwards afterwards
        thisTarget = Variable(
            torch.from_numpy(np.stack([ targetTrain[index] for index in indices ])))
    
        # calculate loss
        loss = lossFunc.forward(output, thisTarget)
        # loss = lossFunc(output, thisTarget)
        
        # accumulate 
        trainLoss += loss
        
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
    # testLoss = lossFunc.forward(output, testTargetVar).data  
    testLoss = lossFunc(output,testTargetVar)
    testLosses.append(testLoss)
    
    print("epoch",epoch,"train loss=", trainLoss.data, "test loss=",testLoss.data)



model.eval()

out = model.forward(inputTest).data.numpy() 

# acutal classification: idea
predictions =[]
for i in out:
    if i>=0.5: predictions.append(1)
    else: predictions.append(0)


#plotting the loss
trainLosses = [trainLosses[i].detach().numpy() for i,_ in enumerate(trainLosses)]
testLosses = [testLosses[i].detach().numpy() for i,_ in enumerate(testLosses)]

plt.plot(trainLosses)
plt.plot(testLosses)
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(["train loss", "test loss"])
plt.show()


# true_pred = testTargetVar.numpy()
# conf_matrix = confusion_matrix(true_pred,predictions)
# print("Confusion Matrix of the Test Set")
# print("-----------")
# print(conf_matrix)
# print("Accuracy :\t"+str(accuracy_score(true_pred,predictions)))
# print("Precision :\t"+str(precision_score(true_pred,predictions)))
# print("Recall :\t"+str(recall_score(true_pred,predictions)))
# print("F1 Score :\t"+str(f1_score(true_pred,predictions)))
    

