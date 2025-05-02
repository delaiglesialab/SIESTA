# Third-Party Imports
import pandas as pd
import numpy as np

mapping = {1: "WAKE", 2: "NREM", 3: "REM", 255.0: 255,
    "wake": "WAKE", "rem": "REM", "nrem": "NREM",
    "awake": "WAKE", "AWAKE": "WAKE", "non rem": "NREM", "non-rem": "NREM",
    "Wake": "WAKE", "Non REM": "NREM",
    "Wake X": "WAKE", "Non REM X": "NREM", "REM X": "REM", "Wake ": "WAKE", "REM ": "REM"
}

def normalizeScore(score):
    if isinstance(score, str):
        return score.lower().strip()
    return score


def incremental_training(X, Y, model, chunk_size=1000, classes=None):
    n_samples = X.shape[0]

    if classes is None:
        classes = np.unique(Y)

    for i in range(0, n_samples, chunk_size):
        X_batch = X[i:i + chunk_size]
        Y_batch = Y[i:i + chunk_size]

        Y_batch = [label for label in Y_batch if label in classes]

        if i == 0:
            model.partial_fit(X_batch, Y_batch, classes=classes)
        else:
            model.partial_fit(X_batch, Y_batch)

    return model


def fitModel(data_values, model_wake, model_sleep, scaler, chunk_size=10000):
    Xtrain = data_values[:, :-2]
    Ytrain = data_values[:, -1].astype(str)

    Xtrain = scaler.transform(Xtrain)
    Ytrain = [mapping.get(normalizeScore(y), y) for y in Ytrain]

    valid_classes = ['WAKE', 'NREM', 'REM']
    mask = np.isin(Ytrain, valid_classes)
    Xtrain = Xtrain[mask]
    Ytrain = np.array(Ytrain)[mask]

    Ytrain_wake = np.where(Ytrain == "WAKE", "WAKE", "SLEEP")
    Xtrain_sleep = Xtrain[Ytrain_wake == "SLEEP"]
    Ytrain_sleep = Ytrain[Ytrain_wake == "SLEEP"]

    model_wake = incremental_training(Xtrain, Ytrain_wake, model_wake, chunk_size, classes=['WAKE', 'SLEEP'])
    model_sleep = incremental_training(Xtrain_sleep, Ytrain_sleep, model_sleep, chunk_size, classes=['NREM', 'REM'])

    return model_wake, model_sleep


def getPredictions(X, model, getProb = False):
    timestamps = X.iloc[:, -2]
    useRows = X.iloc[:, -1]
    X = pd.DataFrame(model['scaler'].transform(X.iloc[:,:-2].values))

    # Step 1
    predictions = pd.DataFrame(index=X.index, columns=['score'])
    clean_indices = X.loc[useRows].index

    predictions.loc[clean_indices] = pd.DataFrame(model['model']['Step 1'].predict(X.dropna())).values

    if getProb:
        probWake = pd.DataFrame(index=X.index, columns=model['model']['Step 1'].classes_, dtype=float)
        probSleep = pd.DataFrame(index=X.index, columns=model['model']['Step 2'].classes_, dtype=float)
        probWake.loc[clean_indices, :] = pd.DataFrame(model['model']['Step 1'].predict_proba(X.dropna())).values

    # Step 2
    if 'SLEEP' in predictions['score'].values:
        sleep_indices = predictions[predictions['score'] == 'SLEEP'].index
        X_sleep = X.loc[sleep_indices, :]

        predictions.loc[sleep_indices] = pd.DataFrame(model['model']['Step 2'].predict(X_sleep)).values
        if getProb:
            probSleep.loc[sleep_indices, :] = pd.DataFrame(model['model']['Step 2'].predict_proba(X_sleep)).values
            probs = pd.concat([probWake, probSleep], axis=1)
            return predictions, probs[['WAKE', 'SLEEP', 'NREM', 'REM']]

    predictions['Timestamps'] = timestamps

    return predictions