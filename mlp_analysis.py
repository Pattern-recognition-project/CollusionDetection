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
    ## Ready the dataset (without auction type).
    data_raw = Data("./DB_Collusion_All_processed.csv")
    df = data_raw.load_aggegrated(data_type='numpy', add_labels=True, min_bids=2)[:,:-2]

    # scale each numeric predictor to have mean of 0 and st. deviation of 1.
    scaler = preprocessing.StandardScaler().fit(df[:,1:19])
    df[:,1:19] = scaler.transform(df[:,1:19])

    x_train, x_test, y_train, y_test = train_test_split(df[:,1:],
                                                        df[:,0],
                                                        test_size=0.2,
                                                        random_state=42)

    # get data ready for models.
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Define the MLP model
    DROPOUT = 0.0
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(agg_data_train.shape[1])))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    model.summary()


    # print(f"start training ^ with traindata of shape: {trainData.shape} and trainlabels of shape: {trainLabels.shape}")

    history = model.fit(
        x_train,
        y_train,
        epochs=30,
        verbose=1,
        batch_size=BATCH_SIZE,
        validation_)

    PlotResults(history, validation=True)

    y_test_predict = model.predict(x_test)
    y_test_predict = np.array([1 if x >= 0.5 else 0 for x in y_test_predict.flatten()])
    y_train_predict = model.predict(x_train)
    y_train_predict = np.array([1 if x >= 0.5 else 0 for x in y_train_predict.flatten()])

    train_report = classification_report(y_train, y_train_predict)
    test_report = classification_report(y_test, y_test_predict)

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
