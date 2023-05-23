# נדב ישי - 20611989

import numpy as np
from numpy import issubdtype
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

#-------------------------- manage all the normalization methods

class Normalization:
    def __init__(self, dataset):
        self.dataset = dataset

    #-------------------------- Z Score normalization
    def ZScore(self):                                               # Z-Score by Scipy
        for (columnName, columnData) in self.dataset.iteritems():
            if issubdtype(self.dataset[columnName].dtype, np.number):
                self.dataset[columnName] = zscore(self.dataset[columnName])

    #-------------------------- Min - Max normalization
    def MinMax(self):                                               # Min-Max by Sklearn
        scaler = MinMaxScaler()
        list = []

        for (columnName, columnData) in self.dataset.iteritems():
            if issubdtype(self.dataset[columnName].dtype, np.number):
                list.append(columnName)

        self.dataset[list] = scaler.fit_transform(self.dataset[list])
        
    #-------------------------- Decimal Scaling normalization
    def DecimalScaling(self):                                       # Our decimal scaling
        maximum = 0
        for (columnName, columnData) in self.dataset.iteritems():
            if issubdtype(self.dataset[columnName].dtype, np.number):
                maximum = int(abs(max(self.dataset[columnName])))
                max_size = len(str(maximum))
                for i in range(len(columnData)):
                    result = columnData[i] / (10 ** max_size)
                    self.dataset.at[i, columnName] = result







