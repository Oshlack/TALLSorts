#=======================================================================================================================
#
#   TALLSorts v0 - Working with the model that contains subtype infomation
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
import pandas as pd

''' --------------------------------------------------------------------------------------------------------------------
Classes
---------------------------------------------------------------------------------------------------------------------'''

class SubtypeClass:
    def __init__(self, label, parent=None, clf=None, scaler=None):
        self.parent = parent
        self.label = label
        self.children = []
        self.clf = clf
        self.scaler = scaler
        if parent is None:
            self.level = 1
        else:
            self.level = parent.level + 1
            if self not in parent.children:
                parent.children.append(self)

    def __str__(self):
        return self.label
    
    def runTest(self, X_test):
        results = {}
        results['y_pred_int'] = self.clf.predict(X_test)
        results['proba'] = self.clf.predict_proba(X_test)
        return results
    
    def deconstruct(self):
        return {'clf':self.clf, 'scaler':self.scaler}

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''

def reconstructSubtypeObj(subtypeObjectDeconstructed, hierarchy):
    nextLevel = hierarchy.copy()
    subtypeObjects = {}
    curLevel = 1
    while nextLevel:
        nextnextLevel = {}
        for label in nextLevel.keys():
            if nextLevel[label][1] == curLevel:
                if nextLevel[label][0] is None:
                    parent = None
                else:
                    parent = subtypeObjects[nextLevel[label][0]]
                subtypeObjects[label] = SubtypeClass(label, parent=parent, 
                                                     clf=subtypeObjectDeconstructed[label]['clf'],
                                                     scaler=subtypeObjectDeconstructed[label]['scaler'])
            else:
                nextnextLevel[label] = nextLevel[label]
        nextLevel = nextnextLevel.copy()
        curLevel += 1
    return subtypeObjects