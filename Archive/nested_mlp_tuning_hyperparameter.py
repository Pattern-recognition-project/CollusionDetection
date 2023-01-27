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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from torch_model_G import Net
import random
import os
from training_function import training_function
from nested_mlp import binary_output
import torch.optim as optim
from ray import tune
import optuna

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


# def tune_model(config):
#     model, output, trainLosses, testLosses= training_function(config["batchSize"], 10, inputTrain, targetTrain, inputTest, targetTest)#, config['lr'])
#     output = binary_output(output)

#     # print(output)
#     acc = accuracy_score(targetTest, output)
#     tune.report(mean_accuracy=acc)

# config = {"batchSize" : tune.grid_search([4,8, 12,16,20,24, 28, 32, 40, 52, 64])} #, "lr": tune.grid_search([0.00001,0.0001,0.001,0.01,0.1])}
# analysis = tune.run(tune_model, config=config)

# print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

# # Get a dataframe for analyzing trial results.
# df = analysis.dataframe()
# df.to_csv("Tuned_parameters.csv")


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
# added_features = data.load_aggegrated()
added_featuresTrain = data.get_agg_train()
added_featuresTest = data.get_agg_test()
scaler = StandardScaler()
added_featuresTrain = scaler.fit_transform(added_featuresTrain)
added_featuresTest = scaler.transform(added_featuresTest)

def objective(trial):

    lr = trial.suggest_float("lr", 0.00001, 0.1, log=True)
    model, output, trainLosses, testLosses= training_function(32, 60, inputTrain, targetTrain, inputTest, targetTest,added_featuresTrain, added_featuresTest,lr)

    # output = binary_output(output)
    # targetTest_int = data.get_test_y().astype(int)
    targetTest_tensor = Variable(torch.from_numpy(np.stack(targetTest)))
    BCE = nn.BCELoss()

    l = BCE(output, targetTest_tensor)

    return l


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

found_param = study.trials_dataframe()
found_param.to_csv(f"tuning_params.csv")

print(study.best_trial)




