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
from TALLSorts.stages.subtype_class import SubtypeClass, reconstructSubtypeObj
from TALLSorts.stages.scaling import scaleForTesting

''' External '''
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

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
    __________
    fit(X, y)
        ***NOT IMPLEMENTED YET: TRAINING*** With supplied training counts and labels, as transformed through the TALLSorts pipeline, fit/train the
        hierarchical classifier. 
        Note: These should be distinct from any samples you wish to validate the result with.

        If y is not supplied, this will simply transform all previous transformers in the TALLSorts pipeline.

    predict(X)
        With the supplied counts, transformed through the TALLSorts pipeline, return a list of predictions.
        Note that this is not the typical output of a sklearn predict function.

    """
    def __init__(self, tallsorts_model_dict, labelThreshDict=None):
        """
        Initialise the class

        Attributes
        __________
        tallsorts_model_dict: dict
            Contains information about the model setup. Keys:
            'hierarchy': the setup of the hierarchical model. A dict of label:(parent,level) pairs
            'subtypeObjects':objects of SubtypeClass. Contains the LogReg classifier and scaler for that model, along with hierarchical information
        n_jobs : int
            How many threads to use to train the classifier(s) concurrently.
        labelThreshDict : dict
            Dict of thresholds with labels as keys and threshold as values. Currently not used.
        **to add: params for use with the model.
        """
        self.hierarchy = tallsorts_model_dict['hierarchy']
        self.scalers = tallsorts_model_dict['scalers']
        self.clfs = tallsorts_model_dict['clfs']

        self.subtypeObjects = reconstructSubtypeObj(tallsorts_model_dict)

        if labelThreshDict is None:
            self.labelThreshDict = {i:0.5 for i in self.hierarchy}
        else:
            self.labelThreshDict = labelThreshDict

    def __str__(self):
        return('A TALLSorts model object')
    
    def fit(self, X, y):
        return self

    def predict(self, X):
        """
        Runs the classifiers on a sample matrix X

        Parameters
        __________
        X : DataFrame
            Contains standardised logCPM gene counts, with genes as columns and samples as rows

        Returns
        __________
        self
            self.levels is a dict with keys in the form of "Level_levelnum_parent". Contains:
                calls_df : DataFrame
                    A DataFrame with information about the top calls. Columns are: y_highest (highest call); proba_raw; proba_adj; y_pred (predicted call); multi_call (bool)
                    Note that y_pred can be 'Unclassified', but y_highest will always be one of the labels.
                probs_raw_df : DataFrame
                    DataFrame with rows as samples and columns as labels. Entries are the probabilities (raw) for each sample for each label.
                probs_adj_df : DataFrame
                    Same as above, but with adjusted probabilities.
                multi_calls : dict
                    Dict with samples that have multiple predicted subtypes as keys. Each dict value is a list, containing tuples of (label, prob) for each predicted label, arranged in descending order of prob.
        """

        # A results dictionary grouped by the classfier levels
        self.levels = {}

        # iterating the testing process by levels
        for level in range(1, max([self.hierarchy[i][1] for i in self.hierarchy])+1):
            if level == 1:
                unique_parents = ['Level0']
            else:
                unique_parents = set([self.hierarchy[i][0] for i in self.hierarchy if self.hierarchy[i][1]==level])
            # iterating by parent at each level
            for parent_label in unique_parents:
                level_name = f'Level_{level}_{parent_label}'
                self.levels[level_name] = {}

                # narrowing down only to the samples of interest at this level
                if level == 1:
                    X_test = X.copy()
                else:
                    parent = self.subtypeObjects[parent_label]
                    prev_level = f'Level_{level-1}_{parent.parent.label}' if level > 2 else 'Level_1_Level0'
                    samples = self.getSamplesFromCall(parent.label, self.levels[prev_level]['calls_df'], self.levels[prev_level]['multi_calls'])
                    if not samples:
                        continue
                    X_test = X.loc[samples]

                # generating the raw probs dataframe
                probs_raw_df = pd.DataFrame(index=X_test.index)
                # generating the adjusted probs dataframe
                probs_adj_df = pd.DataFrame(index=X_test.index)
                
                # running test
                # THERE IS A BUG HERE RELATING TO HIERARCHIES! NEED TO LIMIT TO PARENTS
                if level == 1:
                    labels_to_test = [i for i in self.hierarchy if self.hierarchy[i][1] == 1]
                else:
                    labels_to_test = [i for i in self.hierarchy if (self.hierarchy[i][0] == parent_label and self.hierarchy[i][1] == level)]
                for label_test in labels_to_test:
                    print(f'Testing {label_test}...')

                    X_scaled = scaleForTesting(X_test, self.scalers[parent_label])
                    label_test_results = self.runTest(X_scaled, self.clfs[label_test]) # this is where the test is run

                    probs_raw_df[label_test] = [i[1] for i in label_test_results['proba']]
                    probs_adj_df[label_test] = probs_raw_df[label_test].apply(lambda x: self.adjustProb(x, self.labelThreshDict[label_test]))

                self.levels[level_name]['probs_raw_df'] = probs_raw_df
                self.levels[level_name]['probs_adj_df'] = probs_adj_df

                # generating the calls dataframes
                self.levels[level_name]['calls_df'], self.levels[level_name]['multi_calls'] = self.genCalls(probs_raw_df, probs_adj_df)
                
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
        """
        Given a specific LogReg classfier, this runs the classifier on the test sample.

        Parameters
        __________
        X : DataFrame
            Contains standardised logCPM gene counts, with genes as columns and samples as rows
        clf : LogisticRegression
            A scikit-learn LogReg classifier model

        Returns
        __________
        results : dict
            A dictionary containing the following keys:           
                y_pred_int : array of 0 or 1 corresponding to classifier outcome
                proba : array of probabilities as determined by the classifier
        """
        results = {}
        results['y_pred_int'] = clf.predict(X)
        results['proba'] = clf.predict_proba(X)
        return results
    
    def genCalls(self, probs_raw_df, probs_adj_df):
        """
        Packages the various outputs from the logistic regression models nicely into a calls and multicalls dataframe

        Parameters
        __________
        probs_raw_df : DataFrame
            DataFrame with rows as samples and columns as labels. Entries are the probabilities (raw) for each sample for each label.
        self.probs_adj_df : DataFrame
            Same as above, but with adjusted probabilities.

        Returns
        __________
        calls_df : DataFrame
            A DataFrame with information about the top calls. Columns are: y_highest (highest call); proba_raw; proba_adj; y_pred (predicted call); multi_call (bool)
            Note that y_pred can be 'Unclassified', but y_highest will always be one of the labels.
        multi_calls : dict
            Dict with samples that have multiple predicted subtypes as keys. Each dict value is a list, containing tuples of (label, prob) for each predicted label, arranged in descending order of prob.
        """

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
    
    def getSamplesFromCall(self, label_test, calls_df, multi_calls):
        # given a label, get a list of all samples that were called that label
        valid_samples = calls_df[(calls_df['multi_call'] == False) & (calls_df['y_highest'] == label_test)].index.to_list()
        valid_samples += [i for i in multi_calls if label_test in [j[0] for j in multi_calls[i]]]
        return valid_samples
    

