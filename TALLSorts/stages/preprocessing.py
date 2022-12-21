#=============================================================================
#
#   TALLSorts v0 - Pre-processing Stage
#   Author: Allen Gu, Breon Schmidt
#   License: MIT
#
#=============================================================================

''' --------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------'''

''' Internal '''
pass

''' External '''
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import conorm

''' --------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------'''

class TMM(BaseEstimator, TransformerMixin):
    """
	A class that generates a normalised training set of counts.

	Based on the Trimmed Means of M-values (TMM) method from Oshlack, Robinson et al (2010)
    Python implementation by Georgy Meshcheryakov through `conorm`: https://pypi.org/project/conorm/

    Transforms a counts matrix into log2 of cpm (counts per million), normalised by TMM.
    """

    def __init__(self):
        pass

    def tmm_norm_factors(self, X):
        return conorm.tmm_norm_factors(X.transpose())

    def log_cpm(self, X, nf):
        cpm =  conorm.cpm(X.transpose(), norm_factors=nf).transpose()
        return np.log2(cpm + 0.5)

    def fit(self, counts):
        self.nf = self.tmm_norm_factors(counts)
        return self

    def transform(self, counts):
        tmm_log_cpm = self.log_cpm(counts, self.nf)

        return tmm_log_cpm

class Preprocessing(BaseEstimator, TransformerMixin):
    """
	A class that generates a standardised counts matrix from log2-cpm counts.

	Standardisation is achieved by calculating the z-score for each gene across the samples.
    """

    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, tmm_counts_logcpm):
        self.scalerVals = {self.scaler.feature_names_in_[i]:[self.scaler.mean_[i], self.scaler.scale_[i]] for i in range(self.scaler.n_features_in_)}
        return self

    def transform(self, tmm_counts_logcpm):
        counts_std = pd.DataFrame(np.zeros((tmm_counts_logcpm.shape[0], self.scaler.n_features_in_)))
        counts_std.index = tmm_counts_logcpm.index
        counts_std.columns = self.scaler.feature_names_in_
        for col in counts_std.columns:
            if col in tmm_counts_logcpm.columns:
                counts_std[col] = (tmm_counts_logcpm[col] - self.scalerVals[col][0]) / self.scalerVals[col][1]
        
        return counts_std