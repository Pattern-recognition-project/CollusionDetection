"""This script runs the 'classic' Regularized Logistic Regression Analysis
on the collusion detection problem.
"""
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
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