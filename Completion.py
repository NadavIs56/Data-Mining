# נדב ישי - 20611989

import pandas as pd
import numpy as np   
from numpy import issubdtype

#-------------------------- Manage the values completion of dataset----------
class Completion:
    def __init__(self, datasets):
        self.train = datasets[0]
        self.test = datasets[1]

#-------------------------- complete by classification function
    def CompleteByClassification(self):
        for (columnName, columnData) in self.train.iteritems():
            if not issubdtype(self.train[columnName].dtype, np.number):             # completion for train file
                self.train[columnName] = self.train[columnName].str.lower()
            if columnName == "class":
                break
            li_missing = self.train.loc[pd.isna(self.train[columnName]), :].index.tolist()  # indexes af all the null slots
            if len(li_missing) > 0:
                for r in li_missing:
                    index = self.train.columns.get_loc("class")
                    classfaction = self.train.iloc[r, index]
                    xt = self.train[self.train['class'] == classfaction].dropna(axis=0)
                    if issubdtype(self.train[columnName].dtype, np.number):
                        result_mean = xt.groupby('class', as_index=False)[columnName].mean()[columnName].tolist()
                        self.train.at[r, columnName] = int(result_mean[0])
                    else:
                        freq = xt[columnName].value_counts().idxmax()
                        self.train.at[r, columnName] = freq

        for (columnName, columnData) in self.test.iteritems():
            if not issubdtype(self.test[columnName].dtype, np.number):              # completion for test file
                self.test[columnName] = self.test[columnName].str.lower()
            if columnName == "class":
                break
            li_missing = self.test.loc[pd.isna(self.test[columnName]),:].index.tolist()  # indexes af all the null slots
            if len(li_missing) > 0:
                for r in li_missing:
                    index = self.test.columns.get_loc("class")
                    classfaction = self.test.iloc[r, index]
                    xt = self.test[self.test['class'] == classfaction].dropna(axis=0)
                    if issubdtype(self.test[columnName].dtype, np.number):
                        result_mean = xt.groupby('class', as_index=False)[columnName].mean()[columnName].tolist()
                        self.test.at[r, columnName] = int(result_mean[0])
                    else:
                        freq = xt[columnName].value_counts().idxmax()
                        self.test.at[r, columnName] = freq

#-------------------------- complete by current data set
    def CompleteByCurrentDataSet(self):
        for (columnName, columnData) in self.train.iteritems():
            if not issubdtype(self.train[columnName].dtype, np.number):                 # completion for train file
                self.train[columnName] = self.train[columnName].str.lower()
            if columnName == "class":
                break
            li_missing = self.train.loc[pd.isna(self.train[columnName]), :].index.tolist()  # indexes af all the null slots
            if len(li_missing) > 0:
                for r in li_missing:
                    if issubdtype(self.train[columnName].dtype, np.number):
                        self.train.at[r, columnName] = int(self.train[columnName].mean())
                    else:
                        freq = self.train[columnName].value_counts().idxmax()
                        self.train.at[r, columnName] = freq

        for (columnName, columnData) in self.test.iteritems():
            if not issubdtype(self.test[columnName].dtype, np.number):                  # completion for test file
                self.test[columnName] = self.test[columnName].str.lower()
            if columnName == "class":
                break
            li_missing = self.test.loc[pd.isna(self.test[columnName]), :].index.tolist()  # indexes af all the null slots
            if len(li_missing) > 0:
                for r in li_missing:
                    if issubdtype(self.test[columnName].dtype, np.number):
                        self.test.at[r, columnName] = int(self.test[columnName].mean())
                    else:
                        freq = self.test[columnName].value_counts().idxmax()
                        self.test.at[r, columnName] = freq