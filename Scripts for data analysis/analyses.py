# %% Setup

import os
import sys
import glob
import time
import datetime
import copy
import numpy as np
import pandas as pd
import joblib

from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier 
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier

from mne.io import read_raw_edf
from scipy import stats

homeDir = os.getcwd()

seed = 7
np.random.seed(seed)

mapping = {1: "WAKE", 2: "NREM", 3: "REM", 255.0: 255,
    "wake": "WAKE", "rem": "REM", "nrem": "NREM",
    "awake": "WAKE", "non rem": "NREM", "non-rem": "NREM",
    "Wake": "WAKE", "Non REM": "NREM",
    "Wake X": "WAKE", "Non REM X": "NREM", "REM X": "REM", "Wake ": "WAKE", "REM ": "REM"
}

modelNames = {'ET': ExtraTreesClassifier(), 
              'LDA': LinearDiscriminantAnalysis(solver = 'svd'), 
              'LR': SGDClassifier(penalty = 'l2', loss='log_loss', max_iter=10000, warm_start = True), 
              'PA' : PassiveAggressiveClassifier(),
              'Per': SGDClassifier(loss='perceptron')}

frontal = ['frontal', 'emg']
parietal = ['parietal', 'emg']
both = ['frontal', 'parietal', 'emg']

def normalizeScore(score):
    if isinstance(score, str):
        return score.lower().strip()
    return score

def getData(lab = 'DLI', strain = 'WT', epoch = 10, channels = frontal, getScores=True, returnStacked = True, autocorr = 0, only3scores = False):
    dataDir = os.path.join(homeDir, 'Data', f'{lab}', f'{strain}')
    secs = '_4s' if epoch == 4 else ''
    
    data = []

    for file in [file.split('_emg')[0] for file in glob.glob(os.path.join(dataDir, f"features{secs}", "*_emg_features*.csv"))]:
        print(os.path.basename(file))
        
        dataframes = []
        
        for channel in channels:
            df = pd.read_csv(file + f"_{channel}_features{secs}.csv")
            timestamps = df.iloc[:, -2]
            useRows = df.iloc[:, -1]
            dataframes.append(df.iloc[:, :-2]) 
            
        features = pd.concat(dataframes, ignore_index=True, axis = 1)
        features['Timestamps'] = timestamps
        features['useRows'] = useRows
        
        if (getScores):
            scores = pd.read_csv(file.replace("features", "scores") + ".csv")
            scores = scores + 1 if lab == 'Ellen' else scores
        
        features, scores = features.iloc[:min(len(features), len(scores))], scores.iloc[:min(len(features), len(scores))]
        scores = scores.iloc[:, 0].apply(lambda score: mapping.get(int(score), score) if isinstance(score, (int, float)) or (isinstance(score, str) and score.isdigit()) else mapping.get(score, score))
        
        features = features.assign(Score=scores)
        features = features[features['Score'] != 255].values
        
        if isinstance(autocorr, int) and autocorr > 0:
            if features.shape[0] <= autocorr:
                continue
            colNum = 3 if getScores else 2
            current_features = features[autocorr:, :-colNum]
            previous_features = [
                features[autocorr - i: features.shape[0] - i, :-colNum]
                for i in range(1, autocorr + 1)
            ]
            features = np.hstack([current_features] + previous_features + [features[autocorr:, -colNum:]])
        
        useRowsIndex = -2 if getScores else -1
        features = features[features[:, useRowsIndex].astype(bool)]
        features = np.delete(features, useRowsIndex, axis=1)
        
        if only3scores: features = features[np.isin(features[:,-1], ['WAKE', 'NREM', 'REM']),:] 
        
        data.append(features)
            
    if returnStacked: data = np.vstack(data)
            
    return data

def getTrainingData(autocorr=0, channels = frontal, modelData = 'limited'):   
    if modelData == 'limited':
        WT = getData(channels = channels, autocorr = autocorr)
        SCN1a = getData(strain='SCN1a', channels = channels, autocorr = autocorr)
        
        train = np.vstack((WT, SCN1a))
    elif modelData == 'full':
        WT = getData(channels = channels, autocorr = autocorr)
        SCN1a = getData(strain='SCN1a', channels = channels, autocorr = autocorr)    
        APPPS1 = getData(strain='APP-PS1', channels = channels, autocorr = autocorr)
        NmsVgat = getData(strain='NmsVgats', channels = channels, autocorr = autocorr)
        WT_5 = getData(strain='5sec', channels = channels, autocorr = autocorr)
        
        if channels == frontal:
            TSE = getData(strain='TSE', channels = channels, autocorr = autocorr)
            train = np.vstack((WT, SCN1a, TSE, APPPS1, NmsVgat, WT_5))
        else:
            train = np.vstack((WT, SCN1a, APPPS1, NmsVgat, WT_5))
    
    dataScaler = MinMaxScaler().fit(train[:, :-2])
    X = train[:, :-2]
    Y = train[:, -1].astype(str)

    return X, Y, dataScaler

def fitModel(Xtrain, Ytrain, scaler, modelName, hierarchical=False):
    Xtrain = scaler.transform(Xtrain)
    Ytrain = normalizeScore(Ytrain)
    Xtrain = Xtrain[np.isin(Ytrain, ['WAKE', 'NREM', 'REM'])]
    Ytrain = Ytrain[np.isin(Ytrain, ['WAKE', 'NREM', 'REM'])]
    
    if hierarchical:
        Ytrain_wake = np.where(Ytrain == "WAKE", "WAKE", "SLEEP")
        Xtrain_sleep = Xtrain[Ytrain_wake == "SLEEP"]
        Ytrain_sleep = Ytrain[Ytrain_wake == "SLEEP"]
        
        step1 = clone(modelNames[modelName]).fit(Xtrain, Ytrain_wake)
        step2 = clone(modelNames[modelName]).fit(Xtrain_sleep, Ytrain_sleep)
        model = {'Step 1': step1, 'Step 2': step2}
    else:
        model = clone(modelNames[modelName]).fit(Xtrain, Ytrain)
    
    return model

def refitModel(Xtrain, Ytrain, Xnew, Ynew, scaler, model, modelName, hierarchical=False, repeatn = 1):
    def preprocess(X, Y):
        X = scaler.transform(X)
        Y = normalizeScore(Y)
        return X[np.isin(Y, ['WAKE', 'NREM', 'REM'])], Y[np.isin(Y, ['WAKE', 'NREM', 'REM'])]
    
    Xtrain, Ytrain = preprocess(Xtrain, Ytrain)
    Xnew, Ynew = preprocess(Xnew, Ynew)
    
    if hierarchical:
        Ytrain_wake = np.where(Ytrain == "WAKE", "WAKE", "SLEEP")
        Ynew_wake = np.where(Ynew == "WAKE", "WAKE", "SLEEP")
        
        Xtrain_sleep, Ytrain_sleep = Xtrain[Ytrain_wake == "SLEEP"], Ytrain[Ytrain_wake == "SLEEP"]
        Xnew_sleep, Ynew_sleep = Xnew[Ynew_wake == "SLEEP"], Ynew[Ynew_wake == "SLEEP"]

        if hasattr(model['Step 1'], 'partial_fit'): model['Step 1'].partial_fit(np.vstack([Xtrain, np.repeat(Xnew, repeatn, axis=0)]), np.hstack([Ytrain_wake, np.repeat(Ynew_wake, repeatn, axis=0)]))
        else: model['Step 1'] = clone(modelNames[modelName]).fit(np.vstack([Xtrain, np.repeat(Xnew, repeatn, axis=0)]), np.hstack([Ytrain_wake, np.repeat(Ynew_wake, repeatn, axis=0)]))
        
        if len(Xnew_sleep) > 0:
            if hasattr(model['Step 2'], 'partial_fit'): model['Step 2'].partial_fit(np.vstack([Xtrain_sleep, np.repeat(Xnew_sleep, repeatn, axis=0)]), np.hstack([Ytrain_sleep, np.repeat(Ynew_sleep, repeatn, axis=0)]))
            else: model['Step 2'] = clone(modelNames[modelName]).fit(np.vstack([Xtrain_sleep, np.repeat(Xnew_sleep, repeatn, axis=0)]), np.hstack([Ytrain_sleep, np.repeat(Ynew_sleep, repeatn, axis=0)]))

    else:
        if hasattr(model, 'partial_fit'): model.partial_fit(np.vstack([Xtrain, np.repeat(Xnew, repeatn, axis=0)]), np.hstack([Ytrain, np.repeat(Ynew, repeatn, axis=0)]))
        else: model = clone(modelNames[modelName]).fit(np.vstack([Xtrain, np.repeat(Xnew, repeatn, axis=0)]), np.hstack([Ytrain, np.repeat(Ynew, repeatn, axis=0)]))
    
    return model

def getPredictions(models, X, hierarchical = False, getProb = False):
    X = pd.DataFrame(X)
    
    if hierarchical:
        # Step 1
        predictions = pd.DataFrame(index=X.index, columns=['score'])
        clean_indicies = X.dropna().index
        predictions.loc[clean_indicies] = pd.DataFrame(models['Step 1'].predict(X.dropna())).values
        
        if getProb: 
            probWake = pd.DataFrame(index=X.index, columns=models['Step 1'].classes_, dtype=float)
            probSleep = pd.DataFrame(index=X.index, columns=models['Step 2'].classes_, dtype=float)
            probWake.loc[clean_indicies, :] = pd.DataFrame(models['Step 1'].predict_proba(X.dropna())).values
        
        # Step 2
        if 'SLEEP' in predictions['score'].values:   
            sleep_indices = predictions[predictions['score'] == 'SLEEP'].index
            X_sleep = X.loc[sleep_indices, :]
        
            predictions.loc[sleep_indices] = pd.DataFrame(models['Step 2'].predict(X_sleep)).values
            if getProb: 
                probSleep.loc[sleep_indices, :] = pd.DataFrame(models['Step 2'].predict_proba(X_sleep)).values
                probs = pd.concat([probWake, probSleep], axis=1)
                return predictions, probs[['WAKE', 'SLEEP', 'NREM', 'REM']]
    else:
        predictions = pd.DataFrame(index=X.index, columns=['score'])
        clean_indicies = X.dropna().index
        predictions.loc[clean_indicies] = pd.DataFrame(models.predict(X.dropna())).values
        
        if getProb: 
            probs = pd.DataFrame(index=X.index, columns=models.classes_, dtype=float)
            probs.loc[clean_indicies, :] = pd.DataFrame(models.predict_proba(X.dropna())).values
            return predictions, probs[['WAKE', 'NREM', 'REM']]
        
    return predictions

# %% Model Comparisons (test score, train score, fit time)
compareModels = False

runALL = False
runALLhyper = False
runHierarchical = False
runHierAuto1 = False
runHierAuto2 = False
runHierAuto3 = False
runTwoHours = True
runTwoHoursAuto = True

if compareModels:
    DLI_WT = getData()
    DLI_SCN1A = getData(strain='SCN1a')
    data = np.vstack((DLI_WT, DLI_SCN1A))
    
def hierarchicalModeling(model, model_name, X, Y, train_idx, test_idx, fold_number, twohours = False):
    if twohours:
        train_idx = np.concatenate([train_idx, np.random.choice(test_idx, size=720, replace=False)])
    
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    
    # Step 1
    step1 = clone(model)
    Y_train_wake = np.where(Y_train == "WAKE", "WAKE", "SLEEP")
    start_time = time.time()
    step1.fit(X_train, Y_train_wake)
    fit_time_wake = time.time() - start_time
    
    predictions = step1.predict(X_test)
    predictions_train = step1.predict(X_train)
    
    sleep_test = (predictions   == "SLEEP")
    sleep_train = (predictions_train == "SLEEP")
    
    # Step 2
    if np.sum(sleep_test) > 0:
        X_train_sleep = X_train[Y_train_wake == "SLEEP"]
        Y_train_sleep = Y_train[Y_train_wake == "SLEEP"]
        
        X_test_sleep = X_test[sleep_test]
        
        step2 = clone(model)
        start_time = time.time()
        step2.fit(X_train_sleep, Y_train_sleep)
        fit_time = fit_time_wake + (time.time() - start_time)

        predictions[sleep_test] = step2.predict(X_test_sleep)
        predictions_train[sleep_train] = step2.predict(X_train[sleep_train])

    test_score = f1_score(Y_test, predictions, average="macro")
    train_score = f1_score(Y_train, predictions_train, average="macro")
    
    print(f"{model_name}, Fold {fold_number}: Fit time {fit_time:.4f}, test accuracy: {test_score:.4f}")

    return test_score, train_score, fit_time

def modelComparison(dataset, resample = False, hyperparams = False, allModels = False, hierarchical = False, autocorrn = 0, twohours = False):
    if allModels:
        if hyperparams:
            models = ({
                     'Logistic Regression' : SGDClassifier(penalty = 'l1', loss='log_loss', max_iter=10000, class_weight = 'balanced'), 
                     'Ridge Regression' : RidgeClassifier(alpha = 1, max_iter = 5000),
                     'Linear Discriminant Analysis' : LinearDiscriminantAnalysis(solver = 'svd', tol = 0.1),
                     'Quadratic Discriminant Analysis' : QuadraticDiscriminantAnalysis(reg_param = 0.01),
                     'Decision Tree' : DecisionTreeClassifier(ccp_alpha=0.001, max_depth=50, min_samples_split=10, min_samples_leaf=5, class_weight = 'balanced'),
                     'Random Forest' : RandomForestClassifier(ccp_alpha=0.001, max_depth=50, min_samples_split=10, min_samples_leaf=5, class_weight = 'balanced'),
                     'Extra Trees Classifier' : ExtraTreesClassifier(ccp_alpha=0.001, max_depth=50, min_samples_split=10, min_samples_leaf=5, class_weight = 'balanced'),
                     'Nearest Centroid' : NearestCentroid(shrink_threshold = 2),
                     'Passive Aggressive Classifier' : PassiveAggressiveClassifier(class_weight = 'balanced'),
                     'Perceptron' : SGDClassifier(loss='perceptron', penalty = 'l1'), 
            })
        else:
            models = ({
                     'Logistic Regression' : SGDClassifier(penalty = 'l2', loss='log_loss', max_iter=10000), 
                     'Ridge Regression' : RidgeClassifier(max_iter = 5000),
                     'Linear Discriminant Analysis' : LinearDiscriminantAnalysis(solver = 'svd'), 
                     'Quadratic Discriminant Analysis' : QuadraticDiscriminantAnalysis(),
                     'K-Nearest Neighbors' : KNeighborsClassifier(),
                     'Decision Tree' : DecisionTreeClassifier(),
                     'Random Forest' : RandomForestClassifier(),
                     'Extra Trees Classifier' : ExtraTreesClassifier(),
                     'Gaussian Naive Bayes' : GaussianNB(), 
                     'Bernoulli Naive Bayes' : BernoulliNB(),
                     'Nearest Centroid' : NearestCentroid(),
                     'Passive Aggressive Classifier' : PassiveAggressiveClassifier(),
                     'Perceptron' : SGDClassifier(loss='perceptron'), 
                     'Multi-layer Perceptron' : MLPClassifier(),
                     'Gradient Boosting' : GradientBoostingClassifier(),
                     'Bagging Classifer' : BaggingClassifier(),
                     'Ada Boost Classifier' : AdaBoostClassifier()
            })
    else:
        models = ({
                 'Extra Trees Classifier' : ExtraTreesClassifier(),
                 'Linear Discriminant Analysis' : LinearDiscriminantAnalysis(solver = 'svd'),
                 'Logistic Regression' : SGDClassifier(penalty = 'l2', loss='log_loss', max_iter=10000), 
                 'Passive Aggressive Classifier' : PassiveAggressiveClassifier(),
                 'Perceptron' : SGDClassifier(loss='perceptron')
        })
    
    X = MinMaxScaler().fit(dataset[:, :-2]).transform(dataset[:, :-2])
    Y = dataset[:, -1].astype(str)
    
    if autocorrn>0:
        X = np.hstack([X[i:len(X) - (autocorrn - i)] for i in range(autocorrn)])
        Y = Y[autocorrn:]
    
    results = []  
    fit_times = []  
    train_scores = []  
    
    for model_name, model in models.items():
        print(f"Running model: {model_name}")
        
        if hierarchical:
            kf = KFold(n_splits=6)
            foldResult = Parallel(n_jobs=6)(
                delayed(hierarchicalModeling)(model, model_name, X, Y, train_idx, test_idx, fold_number, twohours)
                for fold_number, (train_idx, test_idx) in enumerate(kf.split(X), start=1)
            )
            result, train_score, fit_time = zip(*foldResult)
            
            results.append(result)
            train_scores.append(train_score)
            fit_times.append(fit_time)
        else:
            cv_results = cross_validate(model, X, Y, cv=6, verbose=3, n_jobs=6, scoring='f1_macro', return_train_score=True)
            
            results.append(cv_results['test_score'])
            train_scores.append(cv_results['train_score'])
            fit_times.append(cv_results['fit_time'])
        
    results = pd.DataFrame({model_name: scores for model_name, scores in zip(models.keys(), results)})
    train_scores = pd.DataFrame({model_name: scores for model_name, scores in zip(models.keys(), train_scores)})
    fit_times = pd.DataFrame({model_name: times for model_name, times in zip(models.keys(), fit_times)})
        
    return results, train_scores, fit_times

# %%% All Classifier Comparison, no hyperparameter tuning
if runALL & compareModels:
    ALL_test, ALL_train, ALL_fit = modelComparison(data, False, False, True)
    
    ALL_test.to_csv("- Model Selection/Test_all.csv", index=False)
    ALL_train.to_csv("- Model Selection/Train_all.csv", index=False)
    ALL_fit.to_csv("- Model Selection/Fit_all.csv", index=False)

# %%% All Classifier Comparison, hyperparameter tuning
if runALLhyper & compareModels:
    ALL_test_hyper, ALL_train_hyper, ALL_fit_hyper= modelComparison(data, False, True, True)
    
    ALL_test_hyper.to_csv("- Model Selection/Test_all_hyper.csv", index=False)
    ALL_train_hyper.to_csv("- Model Selection/Train_all_hyper.csv", index=False)
    ALL_fit_hyper.to_csv("- Model Selection/Fit_all_hyper.csv", index=False)
    
# %%% Models of interest, Hierarchical
if runHierarchical & compareModels: 
    hier_test, hier_train, hier_time = modelComparison(data, hierarchical = True)
    
    hier_test.to_csv("- Model Selection/Test_hierarchical.csv", index=False)
    hier_train.to_csv("- Model Selection/Train_hierarchical.csv", index=False)
    hier_time.to_csv("- Model Selection/Fit_hierarchical.csv", index=False)
    
# %%% Models of interest, Hierarchical + Auto-correlation (1 epoch)
if runHierAuto1 & compareModels: 
    hierauto_test, hierauto_train, hierauto_time = modelComparison(data, hierarchical = True, autocorrn = 1)
    
    hierauto_test.to_csv("- Model Selection/Test_hierauto.csv", index=False)
    hierauto_train.to_csv("- Model Selection/Train_hierauto.csv", index=False)
    hierauto_time.to_csv("- Model Selection/Fit_hierauto.csv", index=False)
    
# %%% Models of interest, Hierarchical + Auto-correlation (2 epochs)
if runHierAuto2 & compareModels: 
    hierauto2_test, hierauto2_train, hierauto2_time = modelComparison(data, hierarchical = True, autocorrn = 2)
    
    hierauto2_test.to_csv("- Model Selection/Test_hierauto2.csv", index=False)
    hierauto2_train.to_csv("- Model Selection/Train_hierauto2.csv", index=False)
    hierauto2_time.to_csv("- Model Selection/Fit_hierauto2.csv", index=False)
    
# %%% Models of interest, Hierarchical + Auto-correlation (3 epoch)
if runHierAuto3 & compareModels: 
    hierauto3_test, hierauto3_train, hierauto3_time = modelComparison(data, hierarchical = True, autocorrn = 3)
    
    hierauto3_test.to_csv("- Model Selection/Test_hierauto3.csv", index=False)
    hierauto3_train.to_csv("- Model Selection/Train_hierauto3.csv", index=False)
    hierauto3_time.to_csv("- Model Selection/Fit_hierauto3.csv", index=False)

# %%% Models of interest, Hierarchical + 2h from target
if runTwoHours & compareModels: 
    twoh_test, twoh_train, twoh_time = modelComparison(data, hierarchical = True, twohours = True)
    
    twoh_test.to_csv("- Model Selection/Test_hier+2h.csv", index=False)
    twoh_train.to_csv("- Model Selection/Train_hier+2h.csv", index=False)
    twoh_time.to_csv("- Model Selection/Fit_hier+2h.csv", index=False)

# %%% Models of interest, Hierarchical + Auto-correlation (1 epoch) + 2h from target
if runTwoHoursAuto & compareModels: 
    twoautoh_test, twoautoh_train, twoautoh_time = modelComparison(data, hierarchical = True, autocorrn = 1, twohours = True)
    
    twoautoh_test.to_csv("- Model Selection/Test_hierauto+2h.csv", index=False)
    twoautoh_train.to_csv("- Model Selection/Train_hierauto+2h.csv", index=False)
    twoautoh_time.to_csv("- Model Selection/Fit_hierauto+2h.csv", index=False)
    
# %%% Get stage accuracies
WT = getData(returnStacked = False, autocorr = 3)
SCN1a = getData(strain = 'SCN1a', returnStacked = False, autocorr = 3)

data = WT + SCN1a

results = []
for i in range(0,len(data)):
    animal = data[i]
    train = np.vstack(data[:i] + data[i+1:])
    
    scaler = MinMaxScaler().fit(train[:, :-2])
    X = train[:, :-2]
    Y = train[:, -1].astype(str)
    model = fitModel(X, Y, scaler, 'LR', hierarchical=True)

    preds = getPredictions(model, scaler.transform(animal[:, :-2]), hierarchical = True)
    
    Ytest = animal[:, -1].astype(str)
    f1_overall = f1_score(Ytest, preds, average='macro')
    f1_wake = f1_score(
        np.where(Ytest == "WAKE", "WAKE", "SLEEP"), 
        np.where(preds == "WAKE", "WAKE", "SLEEP"), 
        average=None, labels=["WAKE", "SLEEP"]
    )
    f1_sleep = f1_score(
        Ytest[Ytest != 'WAKE'], preds[Ytest != 'WAKE'], 
        average=None, labels=["NREM", "REM"]
    )
    results.append([f1_overall] + list(f1_wake) + list(f1_sleep))
    
finalresult = np.vstack(results)

column_means = np.mean(finalresult, axis=0)
column_se = stats.sem(finalresult, axis=0)

# %% Get sample trace scores
runSampleTrace = False

if runSampleTrace:
    sys.path.append(os.path.join(os.getcwd(), "Data"))
    import featureExtraction
    
    Xtrain, Ytrain, scaler = getTrainingData(3)
    model = fitModel(Xtrain, Ytrain, scaler, 'LR', True)
    
    # Saves predictions for probability adjustment testing
    preds = getPredictions(model, scaler.transform(Xtrain), True, True)
    preds[1].to_csv(f'{homeDir}/Data/training_autocorr3_probs.csv', index=False)
    
    # Handles sample trace predictions
    f = read_raw_edf(homeDir + "\\Data\\DLI\\- Figure, Sample\\sample trace.edf", preload=True)
    epoch = 10
    fs = 400
    
    signal = f.get_data(0).flatten()
    signal = signal[:(len(signal) - (len(signal) % (epoch * fs)))]
    s = np.reshape(signal, (int(len(signal)/(epoch*fs)), epoch * fs))
    useRows = (np.amax(s, axis=1) - np.amin(s, axis=1)) != 0
    
    f = f.filter(1, 70, picks = 0)
    f = f.filter(3, 100, picks = 2)
    
    start_date = f.info['meas_date']
    file_end = start_date + datetime.timedelta(seconds = len(f)/fs)
    
    ecog = featureExtraction.CreateFeaturesDataFrame(useRows, f.get_data(0).flatten()*(10**5), 'ECoG_F', epoch, fs, start_date, file_end)
    emg = featureExtraction.CreateFeaturesDataFrame(useRows, f.get_data(0).flatten()*(10**5), 'EMG', epoch, fs, start_date, file_end)
    
    sample_feats = pd.concat([ecog.iloc[:, :-2], emg.iloc[:, :-2]], axis=1).values
    current_features = sample_feats[3:, :]
    previous_features = [
        sample_feats[3 - i: sample_feats.shape[0] - i, :]
        for i in range(1, 3 + 1)
    ]
    sample_feats = scaler.transform(np.hstack([current_features] + previous_features))
    
    preds = getPredictions(model, sample_feats, True, True)
    
    predictions = pd.concat([pd.DataFrame({'score': ["NA", "NA", "NA"]}), preds[0]])
    probabilities = pd.concat([pd.DataFrame("NA", index=range(3), columns=['WAKE', 'SLEEP', 'NREM', 'REM']), preds[1]], ignore_index=True)
    
    predictions.to_csv(homeDir + "\\Data\\DLI\\- Figure, Sample\\sample_scores_siesta.csv", index=False)
    probabilities.to_csv(homeDir + "\\Data\\DLI\\- Figure, Sample\\sample_probabilities_siesta.csv", index=False)

# %% Model Evaluation, Accuracy
modelEvaluation = True

def processMouseTest(testMouse, X, Y, scaler, model, modelName, hierarchical, twohours):
    Xtest = testMouse[:, :-2]
    Ytest = normalizeScore(testMouse[:, -1].astype(str))

    model = refitModel(X, Y, Xtest[np.arange(720),:], Ytest[np.arange(720)], scaler, model, modelName, hierarchical) if twohours else model

    if not hierarchical:
        predictions = model.predict(scaler.transform(Xtest))
        scoreResult = f1_score(Ytest, predictions, average=None, labels=["WAKE", "NREM", "REM"])
    else:
        predictions = model['Step 1'].predict(scaler.transform(Xtest))
        scoreResult = f1_score(np.where(Ytest == "WAKE", "WAKE", "SLEEP"), predictions, average=None, labels=["WAKE", "SLEEP"])
        
        sleep_test = (predictions == "SLEEP")

        # Step 2
        if np.sum(sleep_test) > 0:
            predictions[sleep_test] = model['Step 2'].predict(scaler.transform(Xtest[sleep_test]))
            
            scoreResult = np.concatenate((scoreResult, f1_score(Ytest[sleep_test], predictions[sleep_test], average=None, labels=["NREM", "REM"])))            
    
    return f1_score(Ytest, predictions, average='macro'), scoreResult

def trainTestModel(trainStrain, testStrain, hierarchical=False, autocorr=0, twohours=False, channels = frontal, modelName = 'LR', n_jobs=1):
    filename = trainStrain + ', ' + testStrain + ' - ' + ("hier" if hierarchical else "") + ("auto" if autocorr else "") + ("2h" if twohours else "")
    electrode = 'FP' if 'frontal' in channels and 'parietal' in channels else 'F' if 'frontal' in channels else 'P' if 'parietal' in channels else ''
    filename = filename + f'{electrode}_'
    print(filename)
    
    train = getData(strain=trainStrain, channels = channels, autocorr = autocorr)
    test = getData(strain=testStrain, returnStacked=False, channels = channels, autocorr = autocorr)
    
    scaler = MinMaxScaler().fit(train[:, :-2])
    X = train[:, :-2]
    Y = train[:, -1].astype(str)
    
    model = fitModel(X, Y, scaler, modelName, hierarchical)

    results = Parallel(n_jobs=n_jobs)(
        delayed(processMouseTest)(testMouse, X, Y, scaler, model, modelName, hierarchical, twohours)
        for testMouse in test
    )
    
    overall = pd.DataFrame([r[0] for r in results], columns=["f1 score"])
    scores = pd.DataFrame([r[1] for r in results], columns=(['WAKE', 'SLEEP', 'NREM', 'REM'] if hierarchical else ['WAKE', 'NREM', 'REM']))

    overall.to_csv(f"- Model Selection/Evaluation/{filename}overall.csv", index=False)
    scores.to_csv(f"- Model Selection/Evaluation/{filename}scores.csv", index=False)

if modelEvaluation:
    # %%% Logistic Regression
    # WT Training/SCN1a Validation
    trainTestModel('WT', 'SCN1a', modelName = 'LR')
    trainTestModel('WT', 'SCN1a', True, modelName = 'LR')
    trainTestModel('WT', 'SCN1a', True, 3, modelName = 'LR')
    trainTestModel('WT', 'SCN1a', True, 3, True, modelName = 'LR')
    trainTestModel('WT', 'SCN1a', True, 3, True, channels = parietal, modelName = 'LR')
    trainTestModel('WT', 'SCN1a', True, 3, True, channels = both, modelName = 'LR')
    
    # SCN1a Training/WT Validation
    trainTestModel('SCN1a', 'WT', modelName = 'LR')
    trainTestModel('SCN1a', 'WT', True, modelName = 'LR')
    trainTestModel('SCN1a', 'WT', True, 3, modelName = 'LR')
    trainTestModel('SCN1a', 'WT', True, 3, True, modelName = 'LR')
    trainTestModel('SCN1a', 'WT', True, 3, True, channels = parietal, modelName = 'LR')
    trainTestModel('SCN1a', 'WT', True, 3, True, channels = both, modelName = 'LR')

# %% Model Testing
runModelTesting = False

def accuracyCalculation(mouse, scaler, twohours, hierarchical, model, modelName, Xtrain, Ytrain, repeatn):
    X = scaler.transform(mouse[:, :-2])
    Y = normalizeScore(mouse[:, -1].astype(str))

    newmodel = copy.deepcopy(model)
    newmodel = refitModel(Xtrain, Ytrain, X[np.arange(720),:], Y[np.arange(720)], scaler, newmodel, modelName, hierarchical, repeatn) if twohours else model

    if hierarchical:
        preds = getPredictions(newmodel, X, hierarchical, False)
        
        f1_overall = f1_score(Y, preds, average='macro')
        f1_wake = f1_score(
            np.where(Y == "WAKE", "WAKE", "SLEEP"), 
            np.where(preds == "WAKE", "WAKE", "SLEEP"), 
            average=None, labels=["WAKE", "SLEEP"]
        )
        f1_sleep = f1_score(
            Y[Y != 'WAKE'], preds[Y != 'WAKE'], 
            average=None, labels=["NREM", "REM"]
        )
        return [f1_overall] + list(f1_wake) + list(f1_sleep)
    else:
        preds = getPredictions(newmodel, X, hierarchical, False)
        
        f1_overall = f1_score(Y, preds, average='macro')
        f1 = f1_score(Y, preds, average=None, labels=["WAKE", "NREM", "REM"])
        return [f1_overall] + list(f1)

def getPredictionAccuracy(models, scaler, Xtrain, Ytrain, lab = 'DLI', strain = 'WT', epoch = 10, channels = frontal, modelName = 'LR', returnStacked = False, autocorr = 3, hierarchical = True, twohours = False, n_jobs = 1, only3scores = False, repeatn = 5):
    if lab != 'FK':
        data = getData(lab = lab, strain=strain, epoch = epoch, channels = channels, returnStacked=returnStacked, autocorr = autocorr, only3scores = only3scores) 
        
        result = Parallel(n_jobs=n_jobs)(delayed(accuracyCalculation)(mouse, scaler, twohours, hierarchical, models, modelName, Xtrain, Ytrain, repeatn) for mouse in data)
        result_array = np.array(result)
        
    else:
        data = getData(lab = lab, strain=strain, epoch = epoch, channels = channels, returnStacked=True, autocorr = autocorr)
        result_array = [accuracyCalculation(data, scaler, twohours, hierarchical, models, modelName, Xtrain, Ytrain, repeatn)]
    
    if hierarchical:
        return pd.DataFrame(result_array, columns=['Overall', 'WAKE', 'SLEEP', 'NREM', 'REM'])
    else:
        return pd.DataFrame(result_array, columns=['Overall', 'WAKE', 'NREM', 'REM'])

if runModelTesting:
    # %%% Frontal electrode
    Xtrain, Ytrain, scaler = getTrainingData(3)
    modelAuto = fitModel(Xtrain, Ytrain, scaler, 'LR', True)
    
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = 'NmsVgats').to_csv("- Model Selection/Evaluation/frontal/NmsVgats.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = 'APP-PS1').to_csv("- Model Selection/Evaluation/frontal/APP-PS1.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = 'TSE').to_csv("- Model Selection/Evaluation/frontal/TSE.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = '5sec').to_csv("- Model Selection/Evaluation/frontal/5sec.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = 'NmsVgats', twohours = True).to_csv("- Model Selection/Evaluation/frontal/NmsVgats+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = 'APP-PS1', twohours = True).to_csv("- Model Selection/Evaluation/frontal/APP-PS1+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = 'TSE', twohours = True).to_csv("- Model Selection/Evaluation/frontal/TSE+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = '5sec', twohours = True).to_csv("- Model Selection/Evaluation/frontal/5sec+2h.csv", index=False)
    
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'FK', strain = 'SCN1a').to_csv("- Model Selection/Evaluation/frontal/FK.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'FK', strain = 'SCN1a', twohours = True).to_csv("- Model Selection/Evaluation/frontal/FK+2h.csv", index=False)
    
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'WT').to_csv("- Model Selection/Evaluation/frontal/SippelWT.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'KO').to_csv("- Model Selection/Evaluation/frontal/SippelKO.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'WT', epoch = 4).to_csv("- Model Selection/Evaluation/frontal/SippelWT_4.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'KO', epoch = 4).to_csv("- Model Selection/Evaluation/frontal/SippelKO_4.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'WT', twohours = True).to_csv("- Model Selection/Evaluation/frontal/SippelWT+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'KO', twohours = True).to_csv("- Model Selection/Evaluation/frontal/SippelKO+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'WT', epoch = 4, twohours = True).to_csv("- Model Selection/Evaluation/frontal/SippelWT_4+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'KO', epoch = 4, twohours = True).to_csv("- Model Selection/Evaluation/frontal/SippelKO_4+2h.csv", index=False)
    
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order1').to_csv("- Model Selection/Evaluation/frontal/Ellen1NEW.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order1', epoch = 4).to_csv("- Model Selection/Evaluation/frontal/Ellen1_4NEW.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order2').to_csv("- Model Selection/Evaluation/frontal/Ellen2NEW.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order2', epoch = 4).to_csv("- Model Selection/Evaluation/frontal/Ellen2_4NEW.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order3').to_csv("- Model Selection/Evaluation/frontal/Ellen3NEW.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order3', epoch = 4).to_csv("- Model Selection/Evaluation/frontal/Ellen3_4NEW.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order1', twohours = True).to_csv("- Model Selection/Evaluation/frontal/Ellen1+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order1', epoch = 4, twohours = True).to_csv("- Model Selection/Evaluation/frontal/Ellen1_4+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order2', twohours = True).to_csv("- Model Selection/Evaluation/frontal/Ellen2+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order2', epoch = 4, twohours = True).to_csv("- Model Selection/Evaluation/frontal/Ellen2_4+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order3', twohours = True).to_csv("- Model Selection/Evaluation/frontal/Ellen3+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order3', epoch = 4, twohours = True).to_csv("- Model Selection/Evaluation/frontal/Ellen3_4+2h.csv", index=False)
    
    # %%% Parietal electrode
    Xtrain, Ytrain, scaler = getTrainingData(3, channels = parietal)
    modelAuto = fitModel(Xtrain, Ytrain, scaler, 'LR', True)
    
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = 'NmsVgats', channels = parietal).to_csv("- Model Selection/Evaluation/parietal/NmsVgats.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = 'APP-PS1', channels = parietal).to_csv("- Model Selection/Evaluation/parietal/APP-PS1.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = '5sec', channels = parietal).to_csv("- Model Selection/Evaluation/parietal/5sec.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = 'NmsVgats', twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/NmsVgats+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = 'APP-PS1', twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/APP-PS1+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, strain = '5sec', twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/5sec+2h.csv", index=False)
    
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'FK', strain = 'SCN1a', channels = parietal).to_csv("- Model Selection/Evaluation/parietal/FK.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'FK', strain = 'SCN1a', twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/FK+2h.csv", index=False)
    
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'WT', channels = parietal).to_csv("- Model Selection/Evaluation/parietal/SippelWT.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'KO', channels = parietal).to_csv("- Model Selection/Evaluation/parietal/SippelKO.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'WT', epoch = 4, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/SippelWT_4.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'KO', epoch = 4, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/SippelKO_4.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'WT', twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/SippelWT+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'KO', twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/SippelKO+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'WT', epoch = 4, twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/SippelWT_4+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'KO', epoch = 4, twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/SippelKO_4+2h.csv", index=False)
    
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'WT', channels = parietal, only3scores = True).to_csv("- Model Selection/Evaluation/parietal/SippelWT_3scores.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'KO', channels = parietal, only3scores = True).to_csv("- Model Selection/Evaluation/parietal/SippelKO_3scores.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'WT', epoch = 4, channels = parietal, only3scores = True).to_csv("- Model Selection/Evaluation/parietal/SippelWT_4_3scores.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Sippel', strain = 'KO', epoch = 4, channels = parietal, only3scores = True).to_csv("- Model Selection/Evaluation/parietal/SippelKO_4_3scores.csv", index=False)
    
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order1', channels = parietal).to_csv("- Model Selection/Evaluation/parietal/Ellen1.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order1', epoch = 4, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/Ellen1_4.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order2', channels = parietal).to_csv("- Model Selection/Evaluation/parietal/Ellen2.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order2', epoch = 4, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/Ellen2_4.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order3', channels = parietal).to_csv("- Model Selection/Evaluation/parietal/Ellen3.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order3', epoch = 4, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/Ellen3_4.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order1', twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/Ellen1+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order1', epoch = 4, twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/Ellen1_4+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order2', twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/Ellen2+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order2', epoch = 4, twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/Ellen2_4+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order3', twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/Ellen3+2h.csv", index=False)
    getPredictionAccuracy(modelAuto, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order3', epoch = 4, twohours = True, channels = parietal).to_csv("- Model Selection/Evaluation/parietal/Ellen3_4+2h.csv", index=False)
    
    # %%% Ellen Data
    Xtrain, Ytrain, scaler = getTrainingData(3)
    Ellen1 = getData(lab = 'Ellen', strain = 'order1', returnStacked = False, autocorr = 3)[0]
    Ellen2 = getData(lab = 'Ellen', strain = 'order2', returnStacked = False, autocorr = 3)[0]
    Ellen3 = getData(lab = 'Ellen', strain = 'order3', returnStacked = False, autocorr = 3)[0]
    Xtrain = np.vstack([Xtrain, np.vstack([Ellen1, Ellen2, Ellen3])[:, :-2]])
    Ytrain = np.hstack([Ytrain, np.vstack([Ellen1, Ellen2, Ellen3])[:, -1].astype(str)])
    
    modelDIL_Ellen = fitModel(Xtrain, Ytrain, scaler, 'LR', True)
    
    getPredictionAccuracy(modelDIL_Ellen, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order1').to_csv("- Model Selection/Evaluation/frontal/DIL_Ellen1.csv", index=False)
    getPredictionAccuracy(modelDIL_Ellen, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order1', epoch = 4).to_csv("- Model Selection/Evaluation/frontal/DIL_Ellen1_4.csv", index=False)
    getPredictionAccuracy(modelDIL_Ellen, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order2').to_csv("- Model Selection/Evaluation/frontal/DIL_Ellen2.csv", index=False)
    getPredictionAccuracy(modelDIL_Ellen, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order2', epoch = 4).to_csv("- Model Selection/Evaluation/frontal/DIL_Ellen2_4.csv", index=False)
    getPredictionAccuracy(modelDIL_Ellen, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order3').to_csv("- Model Selection/Evaluation/frontal/DIL_Ellen3.csv", index=False)
    getPredictionAccuracy(modelDIL_Ellen, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order3', epoch = 4).to_csv("- Model Selection/Evaluation/frontal/DIL_Ellen3_4.csv", index=False)
    
    modelEllen = fitModel(np.vstack([Ellen1, Ellen2, Ellen3])[:, :-2], np.vstack([Ellen1, Ellen2, Ellen3])[:, -1].astype(str), scaler, 'LR', True)
    
    getPredictionAccuracy(modelEllen, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order1').to_csv("- Model Selection/Evaluation/frontal/rat_Ellen1.csv", index=False)
    getPredictionAccuracy(modelEllen, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order1', epoch = 4).to_csv("- Model Selection/Evaluation/frontal/rat_Ellen1_4.csv", index=False)
    getPredictionAccuracy(modelEllen, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order2').to_csv("- Model Selection/Evaluation/frontal/rat_Ellen2.csv", index=False)
    getPredictionAccuracy(modelEllen, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order2', epoch = 4).to_csv("- Model Selection/Evaluation/frontal/rat_Ellen2_4.csv", index=False)
    getPredictionAccuracy(modelEllen, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order3').to_csv("- Model Selection/Evaluation/frontal/rat_Ellen3.csv", index=False)
    getPredictionAccuracy(modelEllen, scaler, Xtrain, Ytrain, lab = 'Ellen', strain = 'order3', epoch = 4).to_csv("- Model Selection/Evaluation/frontal/rat_Ellen3_4.csv", index=False)


# %% Testing Sippel frontal electrode vs. parietal electrode
testSippelData = False

if testSippelData:
    def getSippelData(channel = "frontal"):
        WTdata = glob.glob(os.path.join(os.path.join(homeDir, 'Data', 'Sippel', 'WT'), "features", f"*_{channel}_features*.csv")) 
        KOdata = glob.glob(os.path.join(os.path.join(homeDir, 'Data', 'Sippel', 'KO'), "features", f"*_{channel}_features*.csv")) 
        Sippeldata = WTdata + KOdata
        
        faultyNums = ["0985","0999","1573","1677","1729","1811","2011","2016","3040","3051","3055","3056","3142","3534","3976"]
        data = []
    
        for file in Sippeldata:
            print(os.path.basename(file))
            animalNum = f"{os.path.basename(file)[3:7]}"
            
            features = pd.read_csv(file)
            scores = pd.read_csv(file.split(f'_{channel}')[0].replace("features", "scores") + ".csv")
            
            features, scores = features.iloc[:min(len(features), len(scores))], scores.iloc[:min(len(features), len(scores))]
            scores = scores.iloc[:, 0].apply(lambda score: mapping.get(int(score), score) if isinstance(score, (int, float)) or (isinstance(score, str) and score.isdigit()) else mapping.get(score, score))
            features = features.assign(Score=scores)
            features = features[features['Score'] != 255]
             
            features['animalNum'] = animalNum
            features['isFaulty'] = animalNum in faultyNums
            
            data.append(features)
                
        data = pd.concat(data, ignore_index=True)
                
        return data
    
    sippelFrontal = getSippelData(channel = 'frontal')
    sippelFrontal = sippelFrontal[sippelFrontal['Score'] == 'REM']
    
    np.mean(sippelFrontal[sippelFrontal['isFaulty']==False]['ECoG_F_lowfreq'])
    stats.sem(sippelFrontal[sippelFrontal['isFaulty']==False]['ECoG_F_lowfreq'])    
    np.mean(sippelFrontal[sippelFrontal['isFaulty']==True]['ECoG_F_lowfreq'])
    stats.sem(sippelFrontal[sippelFrontal['isFaulty']==True]['ECoG_F_lowfreq'])
    
    
    sippelParietal = getSippelData(channel = 'parietal')
    sippelParietal = sippelParietal[sippelParietal['Score'] == 'REM']
    
    np.mean(sippelParietal[sippelParietal['isFaulty']==False]['ECoG_P_lowfreq'])
    stats.sem(sippelParietal[sippelParietal['isFaulty']==False]['ECoG_P_lowfreq'])
    np.mean(sippelParietal[sippelParietal['isFaulty']==True]['ECoG_P_lowfreq'])
    stats.sem(sippelParietal[sippelParietal['isFaulty']==True]['ECoG_P_lowfreq'])
    
# %% Compare Rat vs Mouse power spectra
ratFiles = [rf"{homeDir}/Data/Ellen/order1/edf/bobmarley_102819.edf",
            rf"{homeDir}/Data/Ellen/order1/edf/bobmarley_102919.edf",
            rf"{homeDir}/Data/Ellen/order1/edf/cheaptrick_110619.edf",
            rf"{homeDir}/Data/Ellen/order1/edf/cheaptrick_110719.edf",
            rf"{homeDir}/Data/Ellen/order1/edf/neilyoung_111819.edf"]

ellenEEG = []
ellenEMG = []
ellen_scores = []
for file in ratFiles:
    f = read_raw_edf(file, preload=True)
    
    ellenEEG = np.append(ellenEEG, f.get_data(0).flatten())
    ellenEMG = np.append(ellenEMG, f.get_data(2).flatten())
    ellen_scores = np.append(ellen_scores, pd.read_csv(file.replace("/edf", "/scores_4s").replace(".edf", ".csv")).values)
ellen_scores = [i for i in np.vectorize(mapping.get)(ellen_scores + 1) for _ in range(4*250)]
ellen = pd.DataFrame({'ecog': ellenEEG, 'emg': ellenEMG, 'scores': ellen_scores})

mouseFiles = [rf"{homeDir}/Data/DLI/WT/edf/cool.edf",
              rf"{homeDir}/Data/Dli/WT/edf/peterbald.edf",
              rf"{homeDir}/Data/DLI/WT/edf/sad!.edf",
              rf"{homeDir}/Data/DLI/SCN1a/edf/Mouse 2.edf",
              rf"{homeDir}/Data/DLI/SCN1a/edf/Mouse 6.edf"]

dliEEG = []
dliEMG = []
dli_scores = []
for file in mouseFiles:
    f = read_raw_edf(file, preload=True)
    
    dliEEG = np.append(dliEEG, f.get_data(0).flatten())
    dliEMG = np.append(dliEMG, f.get_data(2).flatten())
    dli_scores = np.append(dli_scores, pd.read_csv(file.replace("/edf", "/scores").replace(".edf", ".csv")).values)
dli_scores = [i for i in np.vectorize(mapping.get)(dli_scores) for _ in range(10*400)]
dli = pd.DataFrame({'ecog': dliEEG, 'emg': dliEMG, 'scores': dli_scores})


from scipy.signal import welch

stages = ['WAKE', 'NREM', 'REM', 'WAKE', 'NREM', 'REM']
channel = ['ecog', 'ecog', 'ecog', 'emg', 'emg', 'emg']

for i in range(0,6):
    ellen_frequencies, ellen_Pxx = welch(ellen[ellen['scores'] == stages[i]][channel[i]], fs=250, nperseg=256)
    ellen_psd = pd.DataFrame({'freq': ellen_frequencies, 'pxx': ellen_Pxx / np.sum(ellen_Pxx)})
    ellen_psd["species"] = 'rat'
    dli_frequencies, dli_Pxx = welch(dli[dli['scores'] == stages[i]][channel[i]], fs=400, nperseg=256)
    dli_psd = pd.DataFrame({'freq': dli_frequencies, 'pxx': dli_Pxx / np.sum(dli_Pxx)})
    dli_psd["species"] = 'mouse'
    
    final_psd = pd.concat([ellen_psd, dli_psd], axis=0)
    final_psd.to_csv(f"{homeDir}/- Model Selection/rat vs. mouse/{stages[i]}_{channel[i]}.csv", index=False)
    
    

ratFiles = [rf"{homeDir}/Data/Sippel/WT/edf/gk-1729_wm-day2.edf",
            rf"{homeDir}/Data/Sippel/WT/edf/gk-2011_wm-day1.edf",
            rf"{homeDir}/Data/Sippel/WT/edf/gk-1573_wm-day1.edf",
            rf"{homeDir}/Data/Sippel/KO/edf/gk-3056_wm-day3.edf",
            rf"{homeDir}/Data/Sippel/KO/edf/gk-3978_wm-day1.edf"]


ellenEEG = []
ellenEMG = []
ellen_scores = []
for file in ratFiles:
    f = read_raw_edf(file, preload=True)
    
    ellenEEG = np.append(ellenEEG, f.get_data(1).flatten())
    ellenEMG = np.append(ellenEMG, f.get_data(2).flatten())
    ellen_scores = np.append(ellen_scores, pd.read_csv(file.replace("/edf", "/scores_4s").replace(".edf", ".csv")).values)
ellen_scores = [i for i in np.vectorize(mapping.get)(ellen_scores) for _ in range(4*400)]
ellen = pd.DataFrame({'ecog': ellenEEG, 'emg': ellenEMG, 'scores': ellen_scores})


from scipy.signal import welch

stages = ['WAKE', 'NREM', 'REM', 'WAKE', 'NREM', 'REM']
channel = ['ecog', 'ecog', 'ecog', 'emg', 'emg', 'emg']

for i in range(0,6):
    ellen_frequencies, ellen_Pxx = welch(ellen[ellen['scores'] == stages[i]][channel[i]], fs=400, nperseg=256)
    ellen_psd = pd.DataFrame({'freq': ellen_frequencies, 'pxx': ellen_Pxx / np.sum(ellen_Pxx)})
    ellen_psd["species"] = 'mouse (Sippel)'
    
    final_psd = pd.concat([ellen_psd], axis=0)
    final_psd.to_csv(f"{homeDir}/- Model Selection/rat vs. mouse/{stages[i]}_{channel[i]}2.csv", index=False)    

# %% Get Hypnoactagram data
getHyonoactagramData = True

if getHyonoactagramData:
    sys.path.append(os.path.join(os.getcwd(), "Data"))
    import featureExtraction
    
    ecog1 = 0
    emg = 2
    epoch = 10
    fs = 400
    amp = 10**5
    overwrite = True
    autocorr = 3
    getProb = True
    
    Xtrain, Ytrain, scaler = getTrainingData(autocorr)
    models = fitModel(Xtrain, Ytrain, scaler, 'LR', True)
    
    #folders = ['Het B', 'Het J', 'Het K', 'Het M', 'WT A']
    folders = ['Het J']
    
    for folder in folders:
        dataDir = os.path.join(os.getcwd(), 'Data', 'DLI', '- Figure, Jetlag', folder)
        files = glob.glob(os.path.join(dataDir, "edf", "*.edf"))
        
        for file in files:
            print(file)
            
            if os.path.exists(os.path.join(dataDir, "features", os.path.basename(file)[:-4] + "_frontal_features.csv")) and not overwrite: 
                print(f"Features for {os.path.basename(file)[:-4]} already exists, skipping.")
                break
                
            try:
                f = read_raw_edf(file, preload=True)
            except ValueError as e:
                print(f"Skipping file {file}: {e}")
                break
            except Exception as e:
                print(f"An unexpected error occurred while reading {file}: {e}")
                break
            
            signal = f.get_data(ecog1).flatten()
            signal = signal[:(len(signal) - (len(signal) % (epoch * fs)))]
            s = np.reshape(signal, (int(len(signal)/(epoch*fs)), epoch * fs))
            useRows = (np.amax(s, axis=1) - np.amin(s, axis=1)) != 0
    
            f = f.filter(1, 70, picks = ecog1)
            f = f.filter(3, 100, picks = emg)
            
            if len(f.get_data(emg).flatten()) < epoch * fs: break
            
            start_date = f.info['meas_date']
            file_end = start_date + datetime.timedelta(seconds = len(f)/fs)
            
            ecog1data = featureExtraction.CreateFeaturesDataFrame(useRows, f.get_data(ecog1).flatten()*amp, 'ECoG_F', epoch, fs, start_date, file_end)
            emgdata = featureExtraction.CreateFeaturesDataFrame(useRows, f.get_data(emg).flatten()*amp, 'EMG', epoch, fs, start_date, file_end)
            f.close()
            
            features = pd.concat([ecog1data.iloc[:, :-2],emgdata.iloc[:, :-2]], ignore_index=True, axis = 1)
            features['Timestamps'] = ecog1data.iloc[:, -2]
            features['useRows'] = ecog1data.iloc[:, -1]
            
            features = features.values
    
            if isinstance(autocorr, int) and autocorr > 0:
                if features.shape[0] <= autocorr:
                    continue
                current_features = features[autocorr:, :-2]
                previous_features = [
                    features[autocorr - i: features.shape[0] - i, :-2]
                    for i in range(1, autocorr + 1)
                ]
                features = np.hstack([current_features] + previous_features + [features[autocorr:, -2:]])
            
            features = features[features[:, -1].astype(bool)]
            timestamps = features[:, -2]
            
            X = pd.DataFrame(scaler.transform(np.delete(features, [-1, -2], axis=1)))
            
            predictions = pd.DataFrame(index=X.index, columns=['score'])
            clean_indicies = X.dropna().index
            predictions.loc[clean_indicies] = pd.DataFrame(models['Step 1'].predict(X.dropna())).values
            
            if getProb: 
                probWake = pd.DataFrame(index=X.index, columns=models['Step 1'].classes_, dtype=float)
                probSleep = pd.DataFrame(index=X.index, columns=models['Step 2'].classes_, dtype=float)
                probWake.loc[clean_indicies, :] = pd.DataFrame(models['Step 1'].predict_proba(X.dropna())).values
            
            if 'SLEEP' in predictions['score'].values:   
                sleep_indices = predictions[predictions['score'] == 'SLEEP'].index
                X_sleep = X.loc[sleep_indices, :]
                predictions.loc[sleep_indices] = pd.DataFrame(models['Step 2'].predict(X_sleep)).values
                if getProb: 
                    probSleep.loc[sleep_indices, :] = pd.DataFrame(models['Step 2'].predict_proba(X_sleep)).values
                    probs = pd.concat([probWake, probSleep], axis=1)
                
            predictions['Timestamps'] = timestamps
            
            ecog1data.to_csv(os.path.join(dataDir, "features", os.path.basename(file)[:-4] + "_frontal_features.csv"), index=False)
            emgdata.to_csv(os.path.join(dataDir, "features", os.path.basename(file)[:-4] + "_emg_features.csv"), index=False)
            predictions.to_csv(os.path.join(dataDir, "SIESTA", os.path.basename(file)[:-4] + "_SIESTA.csv"), index=False)
            if getProb:
                probs['Timestamps'] = timestamps
                probs.to_csv(os.path.join(dataDir, "probs", os.path.basename(file)[:-4] + "_probs.csv"), index=False)
                
# %% Save model files
Xtrain, Ytrain, scaler = getTrainingData(3)
modelAuto = fitModel(Xtrain, Ytrain, scaler, 'LR', True)
joblib.dump({'model': modelAuto, 'scaler': scaler}, os.path.join(homeDir, 'Data', 'model_frontal.file'), compress = 3)

Xtrain, Ytrain, scaler = getTrainingData(3, channels = both)
modelAuto = fitModel(Xtrain, Ytrain, scaler, 'LR', True)
joblib.dump({'model': modelAuto, 'scaler': scaler}, os.path.join(homeDir, 'Data', 'model_both.file'), compress = 3)