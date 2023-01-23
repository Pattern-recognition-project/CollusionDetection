"""This script runs the 'classic' Regularized Logistic Regression Analysis
on the collusion detection problem.
"""
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LearningCurveDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib import cm

from data import Data

def crossValidationLR(X,y,runs=10):
    """Takes training data as input, and calculates the optimal C and 'penalty'
    hyperparameters for Regularized Logistic Regression.
    """
    # options.
    c_space = np.logspace(-3,1,100)
    regularizer_space = ['l1','l2']
    K = 8 # folds
    scoring = 'f1'

    # set up pipeline, here just the LR.
    cvArguments = {"cv": model_selection.KFold(n_splits=K, shuffle=True),
                   "scoring": scoring}
    pipeline_parameters = {
        'logreg__C': c_space,
        'logreg__penalty': regularizer_space,
        'logreg__solver': ['liblinear'],
        'logreg__fit_intercept': [False],
        'logreg__tol': [1e-4],
        'logreg__max_iter': [200]}
    pipeline = Pipeline(
        [
            ('logreg', LogisticRegression())
        ], verbose=False
    )

    # calculate CV scores.
    scores = np.empty((len(c_space)*len(regularizer_space), runs))
    for run in range(runs):
        print(f'Run:  {run}')
        tuner = model_selection.GridSearchCV(pipeline, pipeline_parameters,
                                             n_jobs=4,
                                             verbose=True,
                                             **cvArguments).fit(X, y)
        scores[:,run] = tuner.cv_results_['mean_test_score']

    scores_mean = np.mean(scores, axis=1) # averaged over runs.
    best_C = tuner.cv_results_['param_logreg__C'][np.argmax(scores_mean)]
    best_penalty = tuner.cv_results_['param_logreg__penalty'][np.argmax(scores_mean)]

    return best_C, best_penalty, np.max(scores_mean)

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

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    ## Hyperparameter Tuning for C and regularization.
    best_C, best_penalty, best_score = crossValidationLR(x_train, y_train, runs=1)

    print(f'best CV F1-score: {best_score}')
    print(f'Best C value: {best_C} | Best penalty: {best_penalty}')

    ## Train the model.
    LR_model = LogisticRegression(C = best_C,
                                  penalty = best_penalty,
                                  solver='liblinear').fit(x_train, y_train)

    ## Results.
    y_train_predict = LR_model.predict(x_train)
    y_test_predict = LR_model.predict(x_test)

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

    ## Plotting learning curves.
    fig, ax = plt.subplots()
    LearningCurveDisplay.from_estimator(LR_model,
                                        x_train,
                                        y_train,
                                        train_sizes=np.linspace(0.01, 1.0, 1000),
                                        n_jobs=4,
                                        ax=ax,
                                        score_type='both',
                                        #line_kw={'marker':'o'},
                                        cv=8,
                                        score_name='F1-score',
                                        scoring='f1',
                                        std_display_style=None)
