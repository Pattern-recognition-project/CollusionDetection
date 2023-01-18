import os
import numpy as np
import pandas as pd
from scipy.stats import moment
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import variation

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from matplotlib import cm

def crossValidationLR(X,y,runs=3):
    c_space = np.logspace(-1, 0.66, 60)

    cvArguments = {"cv": model_selection.KFold(n_splits=8, shuffle=True),
                   "scoring": "accuracy"}

    pipeline_parameters = {
        'logreg__C': c_space,
        'logreg__penalty': ['l1'],
        'logreg__solver': ['liblinear'],
        'logreg__tol': [1e-4],
        'logreg__max_iter': [200]}

    pipeline = Pipeline(
        [
            ('logreg', LogisticRegression())
        ], verbose=False
    )

    scores = np.empty((len(c_space), runs))
    for run in range(runs):
        print(f'Current run:  {run}')
        tuner = model_selection.GridSearchCV(pipeline, pipeline_parameters,
                                             n_jobs=4,
                                             verbose=True,
                                             **cvArguments)
        tuner.fit(X, y)

        scores[:,run] = tuner.cv_results_['mean_test_score']

    scores_mean = np.mean(scores, axis=1)
    scores_std = np.std(scores, axis=1)

    best_C_val = tuner.cv_results_['param_logreg__C'][np.argmax(scores_mean)]
    best_score_val = np.max(scores_mean)

    best_C = str(round(best_C_val, 3))
    best_score = str(round(best_score_val, 3))

    print(f'best C-parameter value :  {best_C}')
    print(f'best accuracy score (avg over runs):  {best_score}')

    df = pd.DataFrame({'param_space': tuner.cv_results_['param_logreg__C'],
                       'accuracies': scores_mean})

    df.plot(x='param_space', y='accuracies', legend=None)
    plt.fill_between(np.array(df['param_space'], dtype='float64'),
                    scores_mean - scores_std,
                    scores_mean + scores_std,
                    alpha=0.2,
                    color="steelblue")
    plt.xlabel('C', fontstyle='italic')
    plt.ylabel('Mean cross-val accuracy  ('+ str(runs) +'x10)', fontstyle='italic')
    plt.grid()
    plt.title('Tuning C for Regularized LR', fontweight=600)
    plt.axvline(x=best_C_val, color='red', alpha=0.8, linewidth=0.8)
    plt.annotate(best_C, (best_C_val + 0.01, np.min(scores_mean - scores_std)), fontweight='bold')
    plt.annotate(best_score, (best_C_val-0.3, best_score_val), fontstyle='italic')
    plt.show()

    return best_C_val

def crossValidationSVM(X,y,runs=3):
    c_space = np.logspace(-1, 1, 60)

    cvArguments = {"cv": model_selection.KFold(n_splits=8, shuffle=True),
                   "scoring": "accuracy"}

    pipeline_parameters = {
        'svm__C': c_space,
        'svm__kernel': ['rbf'],
        'svm__tol': [1e-3],
        'svm__max_iter': [-1]}

    pipeline = Pipeline(
        [
            ('svm', SVC())
        ], verbose=False
    )

    scores = np.empty((len(c_space), runs))
    for run in range(runs):
        print(f'Current run:  {run}')
        tuner = model_selection.GridSearchCV(pipeline, pipeline_parameters,
                                             n_jobs=4,
                                             verbose=True,
                                             **cvArguments)
        tuner.fit(X, y)

        scores[:,run] = tuner.cv_results_['mean_test_score']

    scores_mean = np.mean(scores, axis=1)
    scores_std = np.std(scores, axis=1)

    best_C_val = tuner.cv_results_['param_svm__C'][np.argmax(scores_mean)]
    best_score_val = np.max(scores_mean)

    best_C = str(round(best_C_val, 3))
    best_score = str(round(best_score_val, 3))

    print(f'best C-parameter value :  {best_C}')
    print(f'best accuracy score (avg over runs):  {best_score}')

    df = pd.DataFrame({'param_space': tuner.cv_results_['param_svm__C'],
                       'accuracies': scores_mean})

    df.plot(x='param_space', y='accuracies', legend=None)
    plt.fill_between(np.array(df['param_space'], dtype='float64'),
                    scores_mean - scores_std,
                    scores_mean + scores_std,
                    alpha=0.2,
                    color="steelblue")
    plt.xlabel('C', fontstyle='italic')
    plt.ylabel('Mean cross-val accuracy  ('+ str(runs) +'x10)', fontstyle='italic')
    plt.grid()
    plt.title('Tuning C for SVM', fontweight=600)
    plt.axvline(x=best_C_val, color='red', alpha=0.8, linewidth=0.8)
    plt.annotate(best_C, (best_C_val + 0.01, np.min(scores_mean - scores_std)), fontweight='bold')
    plt.annotate(best_score, (best_C_val-0.3, best_score_val), fontstyle='italic')
    plt.show()

    return best_C_val

def crossValidationMLP(X,y,runs=1):
    alpha_space = np.logspace(-5, 1, 50)
    layer_space = [(12,),(24,),(6,3),(12,6),(24,12),(48,24,12)]

    cvArguments = {"cv": model_selection.KFold(n_splits=8, shuffle=True),
                   "scoring": "accuracy"}

    pipeline_parameters = {
        'mlp__alpha': alpha_space,
        'mlp__hidden_layer_sizes': layer_space,
        'mlp__max_iter': [400]}

    pipeline = Pipeline(
        [
            ('mlp', MLPClassifier())
        ], verbose=False
    )

    scores = np.empty((len(alpha_space) * len(layer_space), runs))
    for run in range(runs):
        print(f'Current run:  {run}')
        tuner = model_selection.GridSearchCV(pipeline, pipeline_parameters,
                                             n_jobs=4,
                                             verbose=True,
                                             **cvArguments)
        tuner.fit(X, y)

        scores[:,run] = tuner.cv_results_['mean_test_score']

    scores_mean = np.mean(scores, axis=1)
    scores_std = np.std(scores, axis=1)

    best_alpha_val = tuner.cv_results_['param_mlp__alpha'][np.argmax(scores_mean)]
    best_layer_val = tuner.cv_results_['param_mlp__hidden_layer_sizes'][np.argmax(scores_mean)]
    best_score_val = np.max(scores_mean)

    best_alpha = str(round(best_alpha_val, 3))
    best_layer = str(best_layer_val)
    best_score = str(round(best_score_val, 3))

    print(f'best alpha-parameter value :  {best_alpha}')
    print(f'best hidden_layer_sizes-parameter value :  {best_layer}')
    print(f'best accuracy score (avg over runs):  {best_score}')

    return best_alpha_val, best_layer_val

def crossValidationRF(X,y,runs=3):
    max_features_space = np.linspace(0.01, 1.0, num=20)
    max_samples_space = np.linspace(0.1, 1.0, num=20)

    cvArguments = {"cv": model_selection.KFold(n_splits=8, shuffle=True),
                   "scoring": "accuracy"}

    pipeline_parameters = {
        'forest__n_estimators': [300],
        'forest__criterion': ['gini'],
        'forest__min_samples_split': [2],
        'forest__min_samples_leaf': [1],
        'forest__max_features': max_features_space,
        'forest__max_samples': max_samples_space,
        'forest__ccp_alpha': [0.0],
        'forest__bootstrap': [True]}

    pipeline = Pipeline(
        [
            ('forest', RandomForestClassifier())
        ], verbose=False
    )

    scores = np.empty((len(max_features_space) * len(max_samples_space), runs))
    for run in range(runs):
        print(f'Current run:  {run}')
        tuner = model_selection.GridSearchCV(pipeline, pipeline_parameters,
                                             n_jobs=4,
                                             verbose=True,
                                             **cvArguments)
        tuner.fit(X, y)

        scores[:, run] = tuner.cv_results_['mean_test_score']

    scores_mean = np.mean(scores, axis=1)

    best_maxfeatures_val = tuner.cv_results_['param_forest__max_features'][np.argmax(scores_mean)]
    best_maxsamples_val = tuner.cv_results_['param_forest__max_samples'][np.argmax(scores_mean)]
    best_score_val = np.max(scores_mean)

    best_maxfeatures = str(round(best_maxfeatures_val, 3))
    best_maxsamples = str(round(best_maxsamples_val, 3))
    best_score = str(round(best_score_val, 3))

    print(f'best maxfeatures-parameter value:  {best_maxfeatures}')
    print(f'best maxsamples-parameter value:  {best_maxsamples}')
    print(f'best accuracy score:  {best_score}')

    return best_maxsamples_val, best_maxfeatures_val

def moments_logreg():
    # use n_bids, mean, variance and 3-5th central moments as features for
    # unregularized logistic regression
    max_central_moment = 5

    df_processed = {'n_bids': [],'mean': []}
    for i in range(2,max_central_moment+1):
        df_processed['moment' + str(i)] = []
    df_processed['collusive'] = []

    for i in range(n_auctions):
        if not above_1[df_raw['Tender']==i].iloc[0]:
            continue

        auction_view = df[df['Tender'] == i]
        auction_bids = auction_view['Bid_value']

        df_processed['n_bids'].append(auction_view.iloc[0]['Number_bids'])
        df_processed['collusive'].append(auction_view.iloc[0]['Collusive_competitor'])
        df_processed['mean'].append(np.mean(auction_bids))

        for i in range(2,max_central_moment+1):
            df_processed['moment' + str(i)].append(moment(auction_bids,moment=i))

    df_processed = pd.DataFrame(df_processed)

    scaler = preprocessing.StandardScaler().fit(df_processed.iloc[:,0:-1])
    scaled_X = scaler.transform(df_processed.iloc[:,0:-1])

    x_train, x_test, y_train, y_test = train_test_split(scaled_X,
                                                        df_processed.iloc[:,-1],
                                                        test_size=0.2,
                                                        random_state=42)

    LR_01 = LogisticRegression(penalty='none').fit(x_train, y_train)
    LR_01_pred = LR_01.predict(x_train)
    train_results = classification_report(y_train,LR_01_pred)
    test_results = classification_report(y_test,LR_01.predict(x_test))

    return train_results, test_results

def shapestatistics_logreg():
    # use n_bids, mean, variance, skew, kurtosis and coefficient of variation
    # as features for L1-regularized logistic regression
    df_processed = {'n_bids': [],'mean': [],'variance': [],'skew': [],'kurtosis': [], 'cv': []}

    df_processed['collusive'] = []

    for i in range(n_auctions):
        if not above_1[df_raw['Tender']==i].iloc[0]:
            continue

        auction_view = df[df['Tender'] == i]
        auction_bids = auction_view['Bid_value']

        df_processed['n_bids'].append(auction_view.iloc[0]['Number_bids'])
        df_processed['collusive'].append(auction_view.iloc[0]['Collusive_competitor'])
        df_processed['mean'].append(np.mean(auction_bids))
        df_processed['variance'].append(np.var(auction_bids))
        df_processed['skew'].append(skew(auction_bids))
        df_processed['kurtosis'].append(kurtosis(auction_bids))
        df_processed['cv'].append(variation(auction_bids))

    df_processed = pd.DataFrame(df_processed)

    scaler = preprocessing.StandardScaler().fit(df_processed.iloc[:,0:-1])
    scaled_X = scaler.transform(df_processed.iloc[:,0:-1])

    x_train, x_test, y_train, y_test = train_test_split(scaled_X,
                                                        df_processed.iloc[:,-1],
                                                        test_size=0.2,
                                                        random_state=42)

    best_C = crossValidationLR(x_train, y_train)

    LR_01 = LogisticRegression(C=best_C,penalty='l1',
                                solver='liblinear').fit(x_train, y_train)
    LR_01_pred = LR_01.predict(x_train)
    train_results = classification_report(y_train,LR_01_pred)
    test_results = classification_report(y_test,LR_01.predict(x_test))

    return train_results, test_results

def includecountry_logreg():
    # use n_bids, mean, variance, skew, kurtosis and coefficient of variation,
    # as well as the country of the dataset as features for L1-regularized
    # logistic regression

    df_processed = {'n_bids': [],'mean': [],'variance': [],'skew': [],'kurtosis': [], 'cv': []}
    for country in countries:
        df_processed[country] = []

    df_processed['collusive'] = []

    for i in range(n_auctions):
        if not above_1[df_raw['Tender']==i].iloc[0]:
            continue

        auction_view = df[df['Tender'] == i]
        auction_bids = auction_view['Bid_value']
        auction_country = countries[int(auction_view.iloc[0]['Dataset'])]
        df_processed['n_bids'].append(auction_view.iloc[0]['Number_bids'])
        df_processed['collusive'].append(auction_view.iloc[0]['Collusive_competitor'])
        df_processed['mean'].append(np.mean(auction_bids))
        df_processed['variance'].append(np.var(auction_bids))
        df_processed['skew'].append(skew(auction_bids))
        df_processed['kurtosis'].append(kurtosis(auction_bids))
        df_processed['cv'].append(variation(auction_bids))
        df_processed[auction_country].append(1)
        for country in countries:
            if country != auction_country: df_processed[country].append(0)

    df_processed = pd.DataFrame(df_processed)

    scaler = preprocessing.StandardScaler().fit(df_processed.iloc[:,0:-7])
    scaled_X = scaler.transform(df_processed.iloc[:,0:-7])
    df_processed.iloc[:,0:-7] = scaled_X
    x_train, x_test, y_train, y_test = train_test_split(df_processed.iloc[:,0:-1],
                                                        df_processed.iloc[:,-1],
                                                        test_size=0.2,
                                                        random_state=42)

    best_C = crossValidationLR(x_train, y_train)

    LR_01 = LogisticRegression(C=best_C,penalty='l1',
                                solver='liblinear').fit(x_train, y_train)
    LR_01_pred = LR_01.predict(x_train)
    train_results = classification_report(y_train,LR_01_pred)
    test_results = classification_report(y_test,LR_01.predict(x_test))

    return train_results, test_results

def includecountry_svm():
    # use n_bids, mean, variance, skew, kurtosis and coefficient of variation,
    # as well as the country of the dataset as features for a L2-regularized
    # SVM
    df_processed = {'n_bids': [],'mean': [],'variance': [],'skew': [],'kurtosis': [], 'cv': []}
    for country in countries:
        df_processed[country] = []

    df_processed['collusive'] = []

    for i in range(n_auctions):
        if not above_1[df_raw['Tender']==i].iloc[0]:
            continue

        auction_view = df[df['Tender'] == i]
        auction_bids = auction_view['Bid_value']
        auction_country = countries[int(auction_view.iloc[0]['Dataset'])]
        df_processed['n_bids'].append(auction_view.iloc[0]['Number_bids'])
        df_processed['collusive'].append(auction_view.iloc[0]['Collusive_competitor'])
        df_processed['mean'].append(np.mean(auction_bids))
        df_processed['variance'].append(np.var(auction_bids))
        df_processed['skew'].append(skew(auction_bids))
        df_processed['kurtosis'].append(kurtosis(auction_bids))
        df_processed['cv'].append(variation(auction_bids))
        df_processed[auction_country].append(1)
        for country in countries:
            if country != auction_country: df_processed[country].append(0)

    df_processed = pd.DataFrame(df_processed)

    scaler = preprocessing.StandardScaler().fit(df_processed.iloc[:,0:-7])
    scaled_X = scaler.transform(df_processed.iloc[:,0:-7])
    df_processed.iloc[:,0:-7] = scaled_X
    x_train, x_test, y_train, y_test = train_test_split(df_processed.iloc[:,0:-1],
                                                        df_processed.iloc[:,-1],
                                                        test_size=0.2,
                                                        random_state=42)

    best_C = crossValidationSVM(x_train, y_train)

    SVM_01 = SVC(C=best_C,kernel='rbf',tol=1e-3,max_iter=-1).fit(x_train, y_train)
    SVM_01_pred = SVM_01.predict(x_train)
    train_results = classification_report(y_train,SVM_01_pred)
    test_results = classification_report(y_test,SVM_01.predict(x_test))

    return train_results, test_results

def includecountry_mlp():
    # use n_bids, mean, variance, skew, kurtosis and coefficient of variation,
    # as well as the country of the dataset as features for a L2-regularized
    # Multi-layer perceptron
    df_processed = {'n_bids': [],'mean': [],'variance': [],'skew': [],'kurtosis': [], 'cv': []}
    for country in countries:
        df_processed[country] = []

    df_processed['collusive'] = []

    for i in range(n_auctions):
        if not above_1[df_raw['Tender']==i].iloc[0]:
            continue

        auction_view = df[df['Tender'] == i]
        auction_bids = auction_view['Bid_value']
        auction_country = countries[int(auction_view.iloc[0]['Dataset'])]
        df_processed['n_bids'].append(auction_view.iloc[0]['Number_bids'])
        df_processed['collusive'].append(auction_view.iloc[0]['Collusive_competitor'])
        df_processed['mean'].append(np.mean(auction_bids))
        df_processed['variance'].append(np.var(auction_bids))
        df_processed['skew'].append(skew(auction_bids))
        df_processed['kurtosis'].append(kurtosis(auction_bids))
        df_processed['cv'].append(variation(auction_bids))
        df_processed[auction_country].append(1)
        for country in countries:
            if country != auction_country: df_processed[country].append(0)

    df_processed = pd.DataFrame(df_processed)

    scaler = preprocessing.StandardScaler().fit(df_processed.iloc[:,0:-7])
    scaled_X = scaler.transform(df_processed.iloc[:,0:-7])
    df_processed.iloc[:,0:-7] = scaled_X
    x_train, x_test, y_train, y_test = train_test_split(df_processed.iloc[:,0:-1],
                                                        df_processed.iloc[:,-1],
                                                        test_size=0.2,
                                                        random_state=42)

    best_alpha, best_layer = crossValidationMLP(x_train,y_train)

    MLP_01 = MLPClassifier(hidden_layer_sizes=best_layer, activation='relu',alpha=best_alpha,max_iter=400).fit(x_train,y_train)
    MLP_01_pred = MLP_01.predict(x_train)
    train_results = classification_report(y_train,MLP_01_pred)
    test_results = classification_report(y_test,MLP_01.predict(x_test))

    return train_results, test_results

def includecountry_rf():
    # use n_bids, mean, variance, skew, kurtosis and coefficient of variation,
    # as well as the country of the dataset as features for a random forest
    df_processed = {'n_bids': [],'mean': [],'variance': [],'skew': [],'kurtosis': [], 'cv': []}
    for country in countries:
        df_processed[country] = []

    df_processed['collusive'] = []

    for i in range(n_auctions):
        if not above_1[df_raw['Tender']==i].iloc[0]:
            continue

        auction_view = df[df['Tender'] == i]
        auction_bids = auction_view['Bid_value']
        auction_country = countries[int(auction_view.iloc[0]['Dataset'])]
        df_processed['n_bids'].append(auction_view.iloc[0]['Number_bids'])
        df_processed['collusive'].append(auction_view.iloc[0]['Collusive_competitor'])
        df_processed['mean'].append(np.mean(auction_bids))
        df_processed['variance'].append(np.var(auction_bids))
        df_processed['skew'].append(skew(auction_bids))
        df_processed['kurtosis'].append(kurtosis(auction_bids))
        df_processed['cv'].append(variation(auction_bids))
        df_processed[auction_country].append(1)
        for country in countries:
            if country != auction_country: df_processed[country].append(0)

    df_processed = pd.DataFrame(df_processed)

    scaler = preprocessing.StandardScaler().fit(df_processed.iloc[:,0:-7])
    scaled_X = scaler.transform(df_processed.iloc[:,0:-7])
    df_processed.iloc[:,0:-7] = scaled_X
    x_train, x_test, y_train, y_test = train_test_split(df_processed.iloc[:,0:-1],
                                                        df_processed.iloc[:,-1],
                                                        test_size=0.2,
                                                        random_state=42)

    best_maxsamples, best_maxfeatures = crossValidationRF(x_train, y_train)

    RF_01 = RandomForestClassifier(n_estimators=300,
                            criterion='gini',
                            min_samples_split=2,
                            min_samples_leaf=1,
                            max_features=best_maxfeatures,
                            ccp_alpha=0.0,
                            bootstrap=True,
                            max_samples=best_maxsamples).fit(x_train, y_train)

    RF_01_pred = RF_01.predict(x_train)
    train_results = classification_report(y_train,RF_01_pred)
    test_results = classification_report(y_test,RF_01.predict(x_test))

    return train_results, test_results

if __name__ == '__main__':
    # preprocess Dataset
    path = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(path, 'DB_Collusion_All_processed.csv')
    df_raw = pd.read_csv(db_path, header=0)
        # select auctions with more than 1 bids (you need >1 bids for collusion)
    above_1 = df_raw['Number_bids'] > 1
    df = df_raw[above_1].copy()
    n_auctions = len(df['Tender'].unique())
    countries = ['Brazil','Italy','America','Switzerland_GR_SG','Switzerland_Ticino','Japan'] # list index == Dataset number.

    #train_results, test_results = moments_logreg()
    # train_results, test_results = shapestatistics_logreg()
    #train_results, test_results = includecountry_logreg()
    #train_results, test_results = includecountry_svm()
    train_results, test_results = includecountry_mlp()
    #train_results, test_results = includecountry_rf()

    print("TRAIN RESULTS")
    print(train_results)
    print("TEST RESULTS")
    print(test_results)
