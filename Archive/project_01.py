import tensorflow as tf
from tensorflow import GradientTape
import numpy as np
from data import Data
import pickle

print("TensorFlow version:", tf.__version__)

BATCH_SIZE = 1


outputModel = tf.keras.Sequential()
outputModel.add(tf.keras.layers.InputLayer(input_shape = (64), batch_size=BATCH_SIZE))
outputModel.add(tf.keras.layers.Dense(64, activation='relu'))
outputModel.add(tf.keras.layers.Dense(1, activation='relu'))

outputModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())


# custom layer
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CustomLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(CustomLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.matmul(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



loss_tracker = tf.keras.metrics.Mean(name="loss")
mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
class CustomModel(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # forward pass input model
            input_pred = self(x, training=True)
            # forward pass this model

            y_pred = outputModel(input_pred, training=True)

            # compute loss
            loss = tf.keras.losses.mean_squared_error(y, y_pred)

        # compute gradients
        trainable_vars = self.trainable_variables #.append(inputModel.trainable_variables)
        gradients = tape.gradient(loss, trainable_vars)

        # update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # return loss
        loss_tracker.update_state(loss)
        mae_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]


with open("DB_Collusion_All_processed.obj","rb") as filehandler:
    data = pickle.load(filehandler)



input = tf.keras.layers.Input(shape = (1), batch_size=BATCH_SIZE)
x1 = tf.keras.layers.Dense(64, activation='relu')(input)
x2 = tf.keras.layers.Dense(64, activation='relu')(x1)
outputLayer = tf.keras.layers.Dense(1, activation='relu')(x2)
inputModel = tf.keras.Model(inputs=input, outputs=outputLayer, name="inputModel")

inputModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())

trainData = data.get_train_X()
trainLabels = data.get_train_y()


trainData = [[[value] for value in auction] for auction in trainData]
trainLabels = [[value] for value in trainLabels]

lx = [[[1], [2], [3]]]

inputModel.summary()

# inputModel.fit(trainData, trainLabels, epochs=1, batch_size=BATCH_SIZE)
# inputModel.fit(tf.ragged.constant([1, 2, 3], [1, 2]), [1, 2], epochs=1, batch_size=BATCH_SIZE, verbose=1)


# trainStep:
#     for bid in action:
#         output.append(inputmodel.forward(x))
#     output / nBids

#     outputModel.forward(output)
#     computLoss
#     updateWeights


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.MeanSquaredError()

def Train(epochs, trainData, trainLabels):
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for x, y in zip(trainData, trainLabels):
            inputModelPredictions = []
            for bid in x:
                pass
                #forward pass
                # inputModelPredictions.append(inputModel(bid, training=True))
            
            # inputModelOutput = inputModelPredictions / len(x)
            inputModelOutput = bid

            output_prediction = outputModel(inputModelOutput, training=True)
            loss = tf.keras.losses.mean_squared_error(y, output_prediction)

            # Compute gradients
            with GradientTape as tape:
                trainable_vars = inputModel.trainable_variables.append(outputModel.trainable_variables)
                gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # # Optimize the model
            # loss_value, grads = grad(model, x, y)
            # optimizer.apply_gradients(zip(grads, model.trainable_variables))



            # Track progress
            epoch_loss_avg.update_state(loss)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, outputModel(x, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))


Train(epochs=1, trainData=trainData, trainLabels=trainLabels)