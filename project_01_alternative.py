import tensorflow as tf
import pickle
import numpy as np
from data import Data


with open("DB_Collusion_All_processed.obj","rb") as filehandler:
    data = pickle.load(filehandler)

class CustomModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.inputModel = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (1), batch_size=1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
        ])

        self.outputModel = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (64), batch_size=1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='relu')
        ])


    def call(self, inputs):
        allInputs = []
        tf.map_fn(lambda x: allInputs.append(x), inputs)
        allInputs = allInputs/len(inputs)
        
        output = self.outputModel(allInputs)
        return output


model = CustomModel()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())

trainData = data.get_train_X()
trainLabels = data.get_train_y()

# trainData = [[[value] for value in auction] for auction in trainData]
# trainLabels = [[value] for value in trainLabels]

paddedTrainData = tf.keras.preprocessing.sequence.pad_sequences(trainData, padding='post', value=-1)
paddedTrainData = np.asarray(paddedTrainData).astype('float32')
trainLabels = np.asarray(trainLabels).astype('float32')

model.fit(paddedTrainData, trainLabels, epochs=10, batch_size=1, verbose=1)