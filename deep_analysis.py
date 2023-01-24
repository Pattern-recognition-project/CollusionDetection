import numpy as np
import pandas as pd
import random
import os
from sklearn import model_selection
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
from matplotlib import cm

from torch_model_G import Net
from data import Data
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


if __name__ == "__main__":
    seed_torch()

    ## Ready the dataset (without auction type or screening variables).
    data_raw = Data("./DB_Collusion_All_processed.csv")
    df = data_raw.load_aggegrated(data_type='numpy', add_labels=True, min_bids=2)
    columns = np.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]).astype(bool)

    # get bid arrays.
    df_min1 = data_raw.load_aggegrated(data_type='numpy', add_labels=True, min_bids=1)
    bids = data_raw.dataset[[x >= 2 for x in df_min1[:,1]]]
        # scale bids per mean and variance of a particular auction
    bids = [[[(bid - np.mean(auction))/np.std(auction)] for bid in auction] for auction in bids]

    # scale each numeric predictor to have mean of 0 and st. deviation of 1.
    scaler = preprocessing.StandardScaler().fit(df[:,1:19])
    df[:,1:19] = scaler.transform(df[:,1:19])

    x_train, x_test, y_train, y_test = train_test_split(df[:,columns],
                                                        df[:,0],
                                                        test_size=0.2,
                                                        random_state=42)
    x_bids_train, x_bids_test, _, _ = train_test_split(bids,
                                                       df[:,0],
                                                       test_size=0.2,
                                                       random_state=42)

    # get data ready for models.
    y_train = y_train
    y_test = y_test

    model, output,output_train, trainLosses, testLosses= training_function(16,
                                                              50,
                                                              x_bids_train,
                                                              y_train,
                                                              x_bids_test,
                                                              y_test,
                                                              x_train.astype('float32'),
                                                              x_test.astype('float32'),
                                                              0.01)

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
    y_test_predict = binary_output(output)
    y_train_predict = binary_output(output_train)

    train_report = classification_report(y_train.astype(int), y_train_predict)
    test_report = classification_report(y_test.astype(int), y_test_predict)

    ## Plotting performance per nr. of bids.
    bid_amounts = (x_test[:,0]*scaler.scale_[0] + scaler.mean_[0]).astype(int)
    bid_amounts_scores = []
    for bid_amount in np.unique(bid_amounts)[:11]:
        idx = bid_amounts == bid_amount
        bid_amounts_scores.append(f1_score(y_test[idx],y_test_predict[idx]))
    idx = bid_amounts >= 13
    bid_amounts_scores.append(f1_score(y_test[idx],y_test_predict[idx]))

    fig, ax = plt.subplots()
    markerline, _, _ = ax.stem(np.unique(bid_amounts)[:12], bid_amounts_scores, linefmt='grey', markerfmt='D')
    markerline.set_markerfacecolor('none')
    ax.set(xlim=(1,14), xticks=np.arange(2,14),
           ylim=(-0.1,1.1), yticks=np.linspace(0,1,11))
    plt.xlabel('Bids per auction', fontstyle='italic')
    plt.ylabel('F1-score', fontstyle='italic')
    plt.show()

    ## Plotting performance per dataset.
    countries = ['Brazil','Italy','America','Switzerland_GR_SG','Switzerland_Ticino','Japan']
    country_scores = []
    fig, axs = plt.subplots(3, 2, figsize=(4, 6), sharex=True, sharey=True)
    axs = axs.flatten()
    for i,  ax in enumerate(axs):
        idx = x_test[:,i+18].astype(bool)
        ConfusionMatrixDisplay.from_predictions(y_test[idx],y_test_predict[idx],cmap=plt.cm.Greens, ax=axs[i], colorbar=False)
        axs[i].set_title(countries[i],fontsize='small',fontweight='semibold')
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')



    axs = axs.reshape(3,2)
    fig.suptitle('Confusion Matrices per dataset.', fontstyle='italic', fontweight='book')
    fig.supxlabel('Predicted Collusion', fontsize='medium', fontstyle='italic')
    fig.supylabel('True Collusion', fontsize='medium', fontstyle='italic')
    plt.show()
