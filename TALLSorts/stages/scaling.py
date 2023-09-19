#=======================================================================================================================
#
#   TALLSorts v0 - Scaling for testing
#   Author: Allen Gu, Breon Schmidt
#   License: MIT
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

''' Internal '''
pass

''' External '''
from sklearn.preprocessing import StandardScaler
import conorm
import numpy as np
import pandas as pd

''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''
pass

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''

def apply_TMM_CPM(X_raw):
    # Applying TMM then CPM normalisation
    nf = conorm.tmm_norm_factors(X_raw.transpose())
    X_cpm = conorm.cpm(X_raw.transpose(), norm_factors=nf).transpose()
    X_logcpm = np.log2(X_cpm + 0.5)
    return X_logcpm

def scaleForTesting(X_raw, scaler):
    X_logcpm = apply_TMM_CPM(X_raw)

    # # set up scaler values
    scalerVals = {scaler.feature_names_in_[i]:[scaler.mean_[i], scaler.scale_[i]] for i in range(scaler.n_features_in_)}

    X_logcpm_2 = pd.DataFrame(np.zeros((X_logcpm.shape[0], scaler.n_features_in_)))
    X_logcpm_2.index = X_logcpm.index
    X_logcpm_2.columns = scaler.feature_names_in_

    for col in X_logcpm_2.columns:
        if col in X_logcpm.columns:
            X_logcpm_2[col] = X_logcpm[col].copy()
        else:
            X_logcpm_2[col] = scalerVals[col][0]
            
    X_scaled = pd.DataFrame(scaler.transform(X_logcpm_2))
    X_scaled.columns = X_logcpm_2.columns
    X_scaled.index = X_logcpm_2.index
    
    return X_scaled

def createScaler(X_raw):
    X_logcpm = apply_TMM_CPM(X_raw)

    scaler = StandardScaler()
    scaler.fit(X_logcpm)

    return scaler