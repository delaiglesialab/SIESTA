import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Set seed
seed = 7
np.random.seed(seed)

# Model
MLmodel = SGDClassifier(penalty = 'l2', loss='log_loss', max_iter=10000)

# Set path
homeDir = r"/mmfs1/home/aibeck/"

# --- No autocorr ---
 Get data
 file = os.path.join(homeDir, "training_noautocorr.csv")

 training_data = pd.read_csv(file)
 Xtrain = training_data.iloc[:, :-2] 
 Ytrain = training_data.iloc[:, -1] 

 Ytrain_wake = np.where(Ytrain == "WAKE", 1, 0)

 Xtrain_sleep = Xtrain[(Ytrain != "WAKE")].copy()
 Ytrain_sleep = Ytrain[(Ytrain != "WAKE")].map({"NREM": 0, "REM": 1})

# Feature Selection Setup
 cv_strategy = KFold(n_splits=5, shuffle=True, random_state=seed)

 step1forward_results, step1backward_results = [], []
 step2forward_results, step2backward_results = [], []

#  Step 1
 for i in range(5):
     print(f"\n--- Forward Selection Run {i+1} ---")
     sfs_forward = SFS(clone(MLmodel), k_features=50, forward=True, floating=False, verbose=2, scoring='accuracy', cv=cv_strategy, n_jobs=60)
     sfs_forward.fit(Xtrain, Ytrain_wake)
     step1forward_results.append(sfs_forward)
    
     pd.DataFrame.from_dict(sfs_forward.get_metric_dict()).T.to_csv(os.path.join(homeDir, f"step1_forward_selection_run_{i+1}.csv"), index=True)
    
 for i in range(5):
     print(f"\n--- Backward Selection Run {i+1} ---")
     sfs_backward = SFS(clone(MLmodel), k_features=50, forward=False, floating=False, verbose=2, scoring='accuracy', cv=cv_strategy, n_jobs=60)
     sfs_backward.fit(Xtrain, Ytrain_wake)
     step1backward_results.append(sfs_backward)
    
     pd.DataFrame.from_dict(sfs_backward.get_metric_dict()).T.to_csv(os.path.join(homeDir, f"step1_backward_selection_run_{i+1}.csv"), index=True)

#  Step 2
 for i in range(5):
     print(f"\n--- Forward Selection Run {i+1} ---")
     sfs_forward = SFS(clone(MLmodel), k_features=50, forward=True, floating=False, verbose=2, scoring='accuracy', cv=cv_strategy, n_jobs=60)
     sfs_forward.fit(Xtrain_sleep, Ytrain_sleep)
     step2forward_results.append(sfs_forward)
        
     pd.DataFrame.from_dict(sfs_forward.get_metric_dict()).T.to_csv(os.path.join(homeDir, f"step2_forward_selection_run_{i+1}.csv"), index=True)
    
 for i in range(5):
     print(f"\n--- Backward Selection Run {i+1} ---")
     sfs_backward = SFS(clone(MLmodel), k_features=50, forward=False, floating=False, verbose=2, scoring='accuracy', cv=cv_strategy, n_jobs=60)
     sfs_backward.fit(Xtrain_sleep, Ytrain_sleep)
     step2backward_results.append(sfs_backward)
    
     pd.DataFrame.from_dict(sfs_backward.get_metric_dict()).T.to_csv(os.path.join(homeDir, f"step2_backward_selection_run_{i+1}.csv"), index=True)
    
# --- 3 epoch time-delay ---
# Get data
file = os.path.join(homeDir, "training_autocorr3.csv")

training_data = pd.read_csv(file)
Xtrain = training_data.iloc[:, :-2] 
Ytrain = training_data.iloc[:, -1] 

Ytrain_wake = np.where(Ytrain == "WAKE", 1, 0)

Xtrain_sleep = Xtrain[(Ytrain != "WAKE")].copy()
Ytrain_sleep = Ytrain[(Ytrain != "WAKE")].map({"NREM": 0, "REM": 1})

#Feature Selection Setup
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=seed)

step1forward_results, step1backward_results = [], []
step2forward_results, step2backward_results = [], []

# Step 1
for i in range(5):
    print(f"\n--- Forward Selection Run {i+1} ---")
    sfs_forward = SFS(clone(MLmodel), k_features=50, forward=True, floating=False, verbose=2, scoring='accuracy', cv=cv_strategy, n_jobs=60)
    sfs_forward.fit(Xtrain, Ytrain_wake)
    step1forward_results.append(sfs_forward)
    
    pd.DataFrame.from_dict(sfs_forward.get_metric_dict()).T.to_csv(os.path.join(homeDir, f"step1_autocorr_forward_selection_run_{i+1}.csv"), index=True)
    
for i in range(5):
    print(f"\n--- Backward Selection Run {i+1} ---")
    sfs_backward = SFS(clone(MLmodel), k_features=50, forward=False, floating=False, verbose=2, scoring='accuracy', cv=cv_strategy, n_jobs=60)
    sfs_backward.fit(Xtrain, Ytrain_wake)
    step1backward_results.append(sfs_backward)
    
    pd.DataFrame.from_dict(sfs_backward.get_metric_dict()).T.to_csv(os.path.join(homeDir, f"step1_autocorr_backward_selection_run_{i+1}.csv"), index=True)

# Step 2
for i in range(5):
    print(f"\n--- Forward Selection Run {i+1} ---")
    sfs_forward = SFS(clone(MLmodel), k_features=50, forward=True, floating=False, verbose=2, scoring='accuracy', cv=cv_strategy, n_jobs=60)
    sfs_forward.fit(Xtrain_sleep, Ytrain_sleep)
    step2forward_results.append(sfs_forward)
        
    pd.DataFrame.from_dict(sfs_forward.get_metric_dict()).T.to_csv(os.path.join(homeDir, f"step2_autocorr_forward_selection_run_{i+1}.csv"), index=True)
    
for i in range(5):
    print(f"\n--- Backward Selection Run {i+1} ---")
    sfs_backward = SFS(clone(MLmodel), k_features=50, forward=False, floating=False, verbose=2, scoring='accuracy', cv=cv_strategy, n_jobs=60)
    sfs_backward.fit(Xtrain_sleep, Ytrain_sleep)
    step2backward_results.append(sfs_backward)
    
    pd.DataFrame.from_dict(sfs_backward.get_metric_dict()).T.to_csv(os.path.join(homeDir, f"step2_autocorr_backward_selection_run_{i+1}.csv"), index=True)