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
    def __init__(self, label, parent=None, clf=None):
        self.parent = parent
        self.label = label
        self.children = []
        self.clf = clf
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
        return {'clf':self.clf}

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''

def reconstructSubtypeObj(tallsorts_model_dict):
    hierarchy = tallsorts_model_dict['hierarchy']
    scalers = tallsorts_model_dict['scalers']
    clfs = tallsorts_model_dict['clfs']

    nextLevel = hierarchy.copy()
    subtypeObjects = {}
    curLevel = 1
    while nextLevel:
        nextnextLevel = {}
        for label in nextLevel:
            if nextLevel[label][1] == curLevel:
                if nextLevel[label][1] == 1:
                    parent = None
                else:
                    parent = subtypeObjects[nextLevel[label][0]]
                subtypeObjects[label] = SubtypeClass(label, parent=parent, 
                                                     clf=clfs[label])
            else:
                nextnextLevel[label] = nextLevel[label]
        nextLevel = nextnextLevel.copy()
        curLevel += 1
    return subtypeObjects

def genSubtypeObjsFromHierarchy(hierarchy):
    subtypeObjects = {}
    for label in hierarchy[hierarchy['Parent'] == ''].index:
        subtypeObjects[label] = SubtypeClass(label, parent=None)
    
    for level in range(2, hierarchy.shape[0]+2):
        remaining = [label for label in hierarchy.index if label not in subtypeObjects]
        if not remaining:
            break
        for label in remaining:
            parent_label = hierarchy.loc[label]['Parent']
            if parent_label in subtypeObjects:
                subtypeObjects[label] = SubtypeClass(label, parent=subtypeObjects[parent_label])

    return subtypeObjects

def gen_hierarchy_dict(subtypeObjects):
    hierarchy_dict = {}
    for label in subtypeObjects:
        obj = subtypeObjects[label]
        if obj.level == 1:
            hierarchy_dict[label] = (None, 1)
        else:
            hierarchy_dict[label] = (obj.parent.label, obj.level)
    return hierarchy_dict