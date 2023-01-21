import tensorflow as tf
import pickle
import numpy as np
import keras_tuner as kt

# import model

from data import Data
from tensorflow_helpers import PlotResults

BATCH_SIZE = 4

with open("DB_Collusion_All_processed.obj","rb") as filehandler:
    data = pickle.load(filehandler)

# load and format the train data
trainData = data.get_train_X()
trainLabels = data.get_train_y()

testData = data.get_test_X()
testLabels = data.get_test_y()

# padded train data
paddedTrainData = tf.keras.preprocessing.sequence.pad_sequences(trainData, maxlen=100, padding='post', value=-1)
paddedTestData = tf.keras.preprocessing.sequence.pad_sequences(testData, maxlen=100, padding='post', value=-1)

# preprocess train data and labels
trainData = np.asarray(paddedTrainData).astype('float32')
trainLabels = np.asarray(trainLabels).astype('float32')
trainData = np.reshape(trainData, (trainData.shape[0], trainData.shape[1], 1))
trainLabels = np.reshape(trainLabels, (trainLabels.shape[0], 1))

mean = np.mean(trainData.flatten())
var = np.var(trainData.flatten())


class GRUHyperModel(kt.HyperModel):
    def model_builder(hp):
        # Define the GRU model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Normalization(axis=None, mean=mean, variance=var))
        model.add(tf.keras.layers.Input(shape=(trainData.shape[1], trainData.shape[2])))
        model.add(tf.keras.layers.Masking(mask_value=-1)) 
        model.add(tf.keras.layers.GRU(hp.Int('GRU units', min_value=32, max_value=512, step=32), input_shape=(trainData.shape[1], trainData.shape[2])))
        model.add(tf.keras.layers.Dense(hp.Int('Dense units', min_value=32, max_value=512, step=32), activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Compile and train the model
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

        return model

# Instantiate the tuner and perform hypertuning
tuner = kt.Hyperband(GRUHyperModel.model_builder, objective='val_loss', max_epochs=10, factor=3)
tuner.search(trainData, trainLabels, epochs=10, validation_split=0.2)

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps)
print(f"Done tuning:\nnumber of GRU units: {best_hps.get('GRU units')}\nnumber of Dense units: {best_hps.get('Dense units')}\nlearning rate: {best_hps.get('learning_rate')}")





# print(f"start training ^ with traindata of shape: {trainData.shape} and trainlabels of shape: {trainLabels.shape}")

history = model.fit(trainData, trainLabels, epochs=10, verbose=1, batch_size=BATCH_SIZE)

# print(f"predicting on: {trainData[4]} and {trainData[5]}")
# print(f"expecting: {trainLabels[4]} and {trainLabels[5]}")

# print(f"shape of traindata4: {trainData[4]} and type is: {type(trainData[4])}")

# print(model.predict(np.array([trainData[4], trainData[5]])))
