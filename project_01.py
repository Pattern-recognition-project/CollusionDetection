import tensorflow as tf
import numpy as np
from data import Data

print("TensorFlow version:", tf.__version__)

BATCH_SIZE = 32

data = Data("./DB_Collusion_All_processed.csv")
print(data.trainX[0])

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = (1), batch_size=BATCH_SIZE))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))

print(model.summary())

outputModel = tf.keras.Sequential()
outputModel.add(tf.keras.layers.InputLayer(input_shape = (64, ), batch_size=BATCH_SIZE))
outputModel.add(tf.keras.layers.Dense(64, activation='relu'))
outputModel.add(tf.keras.layers.Dense(64, activation='relu'))
outputModel.add(tf.keras.layers.Dense(64, activation='relu'))
outputModel.add(tf.keras.layers.Dense(1, activation='relu'))

outputModel.summary()
