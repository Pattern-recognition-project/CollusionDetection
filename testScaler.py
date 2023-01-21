import tensorflow as tf
import pickle
import numpy as np
from data import Data
from math import sqrt

from tensorflow_helpers import PlotResults

BATCH_SIZE = 4

with open("DB_Collusion_All_processed.obj","rb") as filehandler:
    data = pickle.load(filehandler)

# load and format the train data
trainData = data.get_train_X()
# trainLabels = data.get_train_y()

# testData = data.get_test_X()
# testLabels = data.get_test_y()


a = np.hstack(trainData)
mean = np.mean(a)
var = np.var(a)

trainData = [[(x - mean) / sqrt(var) for x in auction] for auction in trainData]


layer = tf.keras.layers.Normalization(axis=None, mean=mean, variance=var)


print(layer(trainData))