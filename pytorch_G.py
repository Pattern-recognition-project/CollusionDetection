# import sys
# sys.settrace

from data import Data
import pickle
import random
import os
import numpy as np

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

from training_function import training_function


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

if __name__ == "__main__":


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


    #train the model
    model, output, trainLosses, testLosses= training_function(4, 10, inputTrain, targetTrain, inputTest, targetTest)


    predictions = binary_output(output)





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

    # metrics 
    true_pred = targetTest.astype(int)
    
    conf_matrix = confusion_matrix(true_pred,predictions)
    print("Confusion Matrix of the Test Set")
    print("-----------")
    print(conf_matrix)
    print("Accuracy :\t"+str(accuracy_score(true_pred,predictions)))
    print("Precision :\t"+str(precision_score(true_pred,predictions)))
    print("Recall :\t"+str(recall_score(true_pred,predictions)))
    print("F1 Score :\t"+str(f1_score(true_pred,predictions)))
        
<<<<<<< HEAD
        optimizer.zero_grad()
    
        # forward through the network
        # inpuTrain[index] is each auction
        # added features for each auction in this batch should be passed as input

        output = model.forward([ inputTrain[index] for index in indices]) #, [added_featuresTrain[index] for index in indices])

    
        # build a PyTorch variable with the target value
        # so that we can propagate backwards afterwards
        thisTarget = Variable(
            torch.from_numpy(np.stack([ targetTrain[index] for index in indices ])))
    
        print('test0')


        # calculate loss
        loss = lossFunc.forward(output, thisTarget)
        # loss = lossFunc(output, thisTarget)

        print('test')
        
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
    
=======
>>>>>>> 1ad8b3683e489f97fbcacbaaf047276051d35f9c

