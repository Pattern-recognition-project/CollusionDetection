import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
from sklearn import preprocessing
import keras_tuner as kt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow_helpers import PlotResults

import matplotlib.pyplot as plt
from matplotlib import cm

from data import Data
BATCH_SIZE = 64

if __name__ == "__main__":
    ## Ready the dataset (without auction type or screening variables).
    data_raw = Data("./DB_Collusion_All_processed.csv")
    df = data_raw.load_aggegrated(data_type='numpy', add_labels=True, min_bids=2)
    columns = np.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]).astype(bool)

    # get bid arrays.
    df_min1 = data_raw.load_aggegrated(data_type='numpy', add_labels=True, min_bids=1)
    bids = data_raw.dataset[[x >= 2 for x in df_min1[:,1]]]
        # scale bids per mean and variance of a particular auction
    bids = np.array([np.array([(bid - np.mean(auction))/np.std(auction) for bid in auction]) for auction in bids], dtype=object)
        # sort bids from low to high
    for i in range(len(bids)):
        bids[i].sort()
        # pad to maximum number of bids
    max_n_bids = np.max([len(auction) for auction in bids])
    bids = np.array([np.pad(bids[i], (0,199-len(bids[i])),mode='constant',constant_values=-100) for i in range(len(bids))])

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
    x_bids_train = x_bids_train.reshape(x_bids_train.shape[0],x_bids_train.shape[1],1)
    x_bids_test = x_bids_test.reshape(x_bids_test.shape[0],x_bids_test.shape[1],1)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # build and train model.
    DROPOUT = 0.5
    bidding_input = tf.keras.layers.Input(shape=(x_bids_train.shape[1], x_bids_train.shape[2]))
    masking = tf.keras.layers.Masking(mask_value=-100)(bidding_input)
    gru = tf.keras.layers.GRU(64, input_shape=(x_bids_train.shape[1], x_bids_train.shape[2]),dropout=DROPOUT, recurrent_dropout=DROPOUT)(masking)
    agg_input = tf.keras.layers.Input(shape=(x_train.shape[1]))
    concat_layer = tf.keras.layers.Concatenate()([agg_input, gru])
    dense = tf.keras.layers.Dense(256, activation='relu')(concat_layer)
    dropout_layer = tf.keras.layers.Dropout(DROPOUT)(dense)
    dense = tf.keras.layers.Dense(128, activation='relu')(dropout_layer)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    model = tf.keras.models.Model(inputs=[bidding_input, agg_input], outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    model.summary()

    history = model.fit([x_bids_train, x_train], y_train, epochs=50, verbose=1, batch_size=BATCH_SIZE, validation_split=0.2)

    PlotResults(history, validation=True)

    y_test_predict = model.predict([x_bids_test, x_test])
    y_test_predict = np.array([1 if x >= 0.5 else 0 for x in y_test_predict.flatten()])
    y_train_predict = model.predict([x_bids_train, x_train])
    y_train_predict = np.array([1 if x >= 0.5 else 0 for x in y_train_predict.flatten()])

    ## Plotting performance per dataset.
    countries = ['Brazil','Italy','America','Switzerland_GR_SG','Switzerland_Ticino','Japan']
    country_scores = []
    fig, axs = plt.subplots(4, 2, figsize=(4, 7), sharex=True, sharey=True)
    axs = axs.flatten()
    for i in range(6):
        idx = x_test[:,i+18].astype(bool)
        ConfusionMatrixDisplay.from_predictions(y_test[idx],y_test_predict[idx],cmap=plt.cm.Greens, ax=axs[i], colorbar=False)
        axs[i].set_title(countries[i],fontsize='small',fontweight='semibold')
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')


    ConfusionMatrixDisplay.from_predictions(y_test,y_test_predict,cmap=plt.cm.Purples, ax=axs[6], colorbar=False)
    axs[6].set_title('All',fontsize='small',fontweight='semibold')
    axs[6].set_xlabel('')
    axs[6].set_ylabel('')


    axs = axs.reshape(4,2)
    fig.suptitle('Confusion Matrices per dataset.', fontstyle='italic', fontweight='book')
    fig.supxlabel('Predicted Collusion', fontsize='medium', fontstyle='italic')
    fig.supylabel('True Collusion', fontsize='medium', fontstyle='italic')
    plt.show()

    train_report = classification_report(y_train, y_train_predict)
    test_report = classification_report(y_test, y_test_predict)
