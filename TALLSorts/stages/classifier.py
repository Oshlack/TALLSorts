#=======================================================================================================================
#
#   TALLSorts v0 - Classifier Stage
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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend  

''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class Classifier(BaseEstimator, ClassifierMixin):
    """
	A class that represents the classifier.

	This classifier actually contains eight separate Logistic Regression classifiers.

	...

	Attributes
	__________
	BaseEstimator : Scikit-Learn BaseEstimator class
		Inherit from this class.
	ClassifierMixin : Scikit-Learn ClassifierMixin class
		Inherit from this class.

	Methods
	-------
	fit(X, y)
		***NOT IMPLEMENTED YET: TRAINING*** With supplied training counts and labels, as transformed through the TALLSorts pipeline, fit/train the
		hierarchical classifier. 
		Note: These should be distinct from any samples you wish to validate the result with.

        If y is not supplied, this will simply transform all previous transformers in the TALLSorts pipeline.

	predict(X)
		With the supplied counts, transformed through the TALLSorts pipeline, return a list of predictions.
        Note that this is not the typical output of a sklearn predict function.

	"""
    def __init__(self, clfDict, label_list, n_jobs=1, labelThreshDict=None):
        """
		Initialise the class

		Attributes
		__________
		clfDict : dict 
			Contains eight LogisticRegression models
        label_list: list
            Contains the labels of the eight subtype classes
        n_jobs : int
			How many threads to use to train the classifier(s) concurrently. Currently a placeholder. For use when training is implemented.
        labelThreshDict : dict
            Dict of thresholds with labels as keys and threshold as values. Currently not used.
        **to add: params for use with the model.
		"""

        self.clfDict = clfDict
        self.label_list = label_list
        self.n_jobs = n_jobs
        if labelThreshDict is None:
            self.labelThreshDict = {i:0.5 for i in self.label_list}
        else:
            self.labelThreshDict = labelThreshDict

    def fit(self, X, y):
        if y is not None:
            # will parallelise this at some point
            def performing_training(label_test):
                y_train = pd.Series(y) == label_test
                logreg = LogisticRegression(random_state=0, max_iter=10000, tol=0.0001, penalty='l1', solver='saga', C=0.2, class_weight='balanced')
                clf = logreg.fit(X, y_train)
                self.clfDict[label_test] = clf
            
            with parallel_backend('threading', n_jobs=self.n_jobs):
                Parallel(verbose=1)(delayed(performing_training)(label) for label in self.label_list)
            
        return self

    def predict(self, X):
        """
		Runs the classifier

		Returns
		__________
        self.calls_df : DataFrame
            A DataFrame with information about the top calls. Columns are: y_highest (highest call); proba_raw; proba_adj; y_pred (predicted call); multi_call (bool)
            Note that y_pred can be 'Unclassified', but y_highest will always be one of the labels.
        self.probs_raw_df : DataFrame
            DataFrame with rows as samples and columns as labels. Entries are the probabilities (raw) for each sample for each label.
        self.probs_adj_df : DataFrame
            Same as above, but with adjusted probabilities.
        self.multi_calls : dict
            Dict with samples that have multiple predicted subtypes as keys. Each dict value is a list, containing tuples of (label, prob) for each predicted label, arranged in descending order of prob.
		"""
        # generating the raw probs dataframe
        probs_raw_df = pd.DataFrame(index=X.index)
        # generating the adjusted probs dataframe
        probs_adj_df = pd.DataFrame(index=X.index)
        
        # running test
        for label_test in self.label_list:
            label_test_results = self.runTest(X, self.clfDict[label_test]) # this is where the test is run
            probs_raw_df[label_test] = [i[1] for i in label_test_results['proba']]
            probs_adj_df[label_test] = probs_raw_df[label_test].apply(lambda x: self.adjustProb(x, self.labelThreshDict[label_test]))
        self.probs_raw_df = probs_raw_df
        self.probs_adj_df = probs_adj_df

        # generating the calls dataframe
        self.calls_df, self.multi_calls = self.genCalls(self.probs_raw_df, self.probs_adj_df)

        return self

    """

    The following functions are used by self.predict

    """
    def adjustProb(self, prob, thresh):
        """
        function to adjust a raw probability using the threshold. Not used if threshold = 0.5
        """
        factor = thresh if prob < thresh else 1-thresh
        return 0.5 + (prob-thresh) * 0.5 / factor

    def runTest(self, X, clf):
        results = {}
        results['y_pred_int'] = clf.predict(X)
        results['proba'] = clf.predict_proba(X)
        return results

    def genCalls(self, probs_raw_df, probs_adj_df):
        calls_df = pd.DataFrame()
        multi_calls = {}
        for sample in probs_adj_df.index:
            sample_probs_raw = probs_raw_df.loc[sample]
            sample_probs_adj = probs_adj_df.loc[sample]
            sample_probs_adj = sample_probs_adj.sort_values(ascending=False)
            sample_probs_raw = sample_probs_raw.loc[sample_probs_adj.index]
            sample_call_df = pd.DataFrame({'y_highest':sample_probs_adj.index[0],
                                        'proba_raw':sample_probs_raw.iloc[0],
                                        'proba_adj':sample_probs_adj.iloc[0]}, index=[sample])
            if sum(sample_probs_adj > 0.5) == 0:
                # no calls were made, return highest non-threshold count
                sample_call_df['y_pred'] = 'Unclassified'
                sample_call_df['multi_call'] = False
            elif sum(sample_probs_adj > 0.5) == 1:
                # exactly one call was made
                sample_call_df['y_pred'] = sample_probs_adj.index[0]
                sample_call_df['multi_call'] = False
            else:
                # more than one call was made
                sample_call_df['y_pred'] = sample_probs_adj.index[0]
                sample_call_df['multi_call'] = True
                multi_call_labels = sample_probs_adj.index[sample_probs_adj > 0.5]
                multi_calls[sample] = [(i, sample_probs_raw.loc[i]) for i in multi_call_labels]
            calls_df = pd.concat([calls_df, sample_call_df])
        return calls_df, multi_calls