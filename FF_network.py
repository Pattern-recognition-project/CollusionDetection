import tensorflow as tf
import pickle
import numpy as np
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
# import model

from data import Data
from tensorflow_helpers import PlotResults
from math import sqrt
BATCH_SIZE = 4

with open("DB_Collusion_All_processed.obj","rb") as filehandler:
    data = pickle.load(filehandler)

agg_data_train = data.get_agg_train()
agg_data_test = data.get_agg_test()

scaler = StandardScaler()

agg_data_train = scaler.fit_transform(agg_data_train)
agg_data_test = scaler.transform(agg_data_test)


trainLabels = data.get_train_y()
testLabels = data.get_test_y()


trainLabels = np.asarray(trainLabels).astype('float32')
trainLabels = np.reshape(trainLabels, (trainLabels.shape[0], 1))

testLabels = np.asarray(testLabels).astype('float32')
testLabels = np.reshape(testLabels, (testLabels.shape[0], 1))




# Define the GRU model
model = tf.keras.Sequential()
# model.add(tf.keras.layers.Normalization(axis=None, mean=mean, variance=var))
model.add(tf.keras.layers.Input(shape=(agg_data_train.shape[1])))
model.add(tf.keras.layers.Dense(24, activation='relu'))
model.add(tf.keras.layers.Dense(24, activation='relu'))
model.add(tf.keras.layers.Dense(24, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()


# print(f"start training ^ with traindata of shape: {trainData.shape} and trainlabels of shape: {trainLabels.shape}")

history = model.fit(
    agg_data_train, 
    trainLabels, 
    epochs=30, 
    verbose=1, 
    batch_size=BATCH_SIZE,
    validation_data=(agg_data_test, testLabels))

PlotResults(history, validation=True)
# print(f"predicting on: {trainData[4]} and {trainData[5]}")
# print(f"expecting: {trainLabels[4]} and {trainLabels[5]}")

# print(f"shape of traindata4: {trainData[4]} and type is: {type(trainData[4])}")

# print(model.predict(np.array([trainData[4], trainData[5]])))
