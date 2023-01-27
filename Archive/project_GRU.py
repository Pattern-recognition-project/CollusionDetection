import tensorflow as tf
import pickle
import numpy as np
import keras_tuner as kt

# import model

from data import Data
from tensorflow_helpers import PlotResults
from math import sqrt
BATCH_SIZE = 4

with open("DB_Collusion_All_processed.obj","rb") as filehandler:
    data = pickle.load(filehandler)

# load and format the train data
trainData = data.get_train_X()
trainLabels = data.get_train_y()

testData = data.get_test_X()
testLabels = data.get_test_y()



a = np.hstack(trainData)
mean = np.mean(a)
var = np.var(a)

trainData = [[(x - mean) / sqrt(var) for x in auction] for auction in trainData]
testData = [[(x - mean) / sqrt(var) for x in auction] for auction in testData]
# padded train data
paddedTrainData = tf.keras.preprocessing.sequence.pad_sequences(trainData, maxlen=100, padding='post', value=-1)
paddedTestData = tf.keras.preprocessing.sequence.pad_sequences(testData, maxlen=100, padding='post', value=-1)

# preprocess train data and labels
trainData = np.asarray(paddedTrainData).astype('float32')
trainLabels = np.asarray(trainLabels).astype('float32')
trainData = np.reshape(trainData, (trainData.shape[0], trainData.shape[1], 1))
trainLabels = np.reshape(trainLabels, (trainLabels.shape[0], 1))



# Define the GRU model
model = tf.keras.Sequential()
# model.add(tf.keras.layers.Normalization(axis=None, mean=mean, variance=var))
model.add(tf.keras.layers.Input(shape=(trainData.shape[1], trainData.shape[2])))
model.add(tf.keras.layers.Masking(mask_value=-1)) 
model.add(tf.keras.layers.GRU(64, input_shape=(trainData.shape[1], trainData.shape[2])))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])



# print(f"start training ^ with traindata of shape: {trainData.shape} and trainlabels of shape: {trainLabels.shape}")

history = model.fit(trainData, trainLabels, epochs=30, verbose=1, batch_size=BATCH_SIZE)

PlotResults(history)
# print(f"predicting on: {trainData[4]} and {trainData[5]}")
# print(f"expecting: {trainLabels[4]} and {trainLabels[5]}")

# print(f"shape of traindata4: {trainData[4]} and type is: {type(trainData[4])}")

# print(model.predict(np.array([trainData[4], trainData[5]])))
