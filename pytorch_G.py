from data import Data
import pickle
import random
import os
import numpy as np

import matplotlib.pyplot as plt
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from torch_model_G import Net
from sklearn.preprocessing import StandardScaler
from training_function import training_function, binary_output


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
    added_features = data.load_aggegrated(data_type="pandas")
    added_featuresTrain = data.get_agg_train()
    added_featuresTest = data.get_agg_test()

    scaler = StandardScaler()
    added_featuresTrain = scaler.fit_transform(added_featuresTrain)#[:,:18]
    added_featuresTest = scaler.transform(added_featuresTest)#[:,:18]


    # get learning rate from tuning
    params = pd.read_csv("tuning_params.csv")
    lr = params.loc[np.argmin(params['value']), 'params_lr']


    # train the model
    # sembra fare meglio con 32 batches rispetto a 4
    model, output, output_train, trainLosses, testLosses= training_function(16, 100, inputTrain, targetTrain, inputTest, targetTest, added_featuresTrain,added_featuresTest, lr)



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
    predictions = binary_output(output)

    true_pred = targetTest.astype(int)

    conf_matrix = confusion_matrix(true_pred,predictions)
    print("Confusion Matrix of the Test Set")
    print("-----------")
    print(conf_matrix)
    print("-----------")

    print("Accuracy :\t"+str(accuracy_score(true_pred,predictions)))
    print("Precision :\t"+str(precision_score(true_pred,predictions)))
    print("Recall :\t"+str(recall_score(true_pred,predictions)))
    print("F1 Score :\t"+str(f1_score(true_pred,predictions)))
