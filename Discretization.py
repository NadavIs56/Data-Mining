# נדב ישי - 20611989

import pandas as pd
import numpy as np
from numpy import issubdtype
import itertools
from math import log
from collections import Counter


class Discretization:
    def __init__(self, dataset, count_bins=2):
        self.count_bins = count_bins
        self.dataset = dataset

    # -------------------------- Pandas Equal frequency discretization
    def PandasEqualFrequencydiscretization(self):
        """ Each bin has the same amount of values """
        for (columnName, columnData) in self.dataset.iteritems():
            if issubdtype(self.dataset[columnName].dtype, np.number):
                self.dataset[columnName] = pd.qcut(self.dataset[columnName],  self.count_bins, duplicates='drop')

    #-------------------------- Our Equal frequency discretization
    def OurEqualFrequencydiscretization(self):
        size, new_lst, count = len(self.dataset.index), [], 0
        dev = self.count_bins
        num = size / dev
        num = int(num)
        column_value = []

        for (columnName, columnData) in self.dataset.iteritems():
            if issubdtype(self.dataset[columnName].dtype, np.number):
                column_value = self.dataset[columnName].copy()
                column_value = column_value.sort_values(ignore_index=True)               # creating a list of all the numeric columns and their values
                disc_list = []
                index = 0
                count_arr = [0] * dev
                count_lastbin = num

                for i in range(dev):
                    temp = []
                    for j in range(num):                         # create bins with equal number of values
                        temp.append(column_value[index])
                        index += 1
                    disc_list.append(temp)

                if index < size:                                # for the last bin
                    for i in range(index, size):
                        count_lastbin += 1
                        disc_list[dev - 1].append(column_value[i])

                index = 0
                new_train = []
                column_set = list(dict.fromkeys(self.dataset[columnName]))
                dic = {}
                for i in column_set:
                    index = 0
                    while (index < dev):                    # building the bins in equal frequency method
                        if i in disc_list[index]:
                            dic[i] = pd.Interval(left=disc_list[index][0], right=disc_list[index][-1])
                            count_arr[index] += 1
                            break
                        else:
                            index += 1
                self.dataset[columnName] = self.dataset[columnName].map(dic)

    #-------------------------- Pandas Equal width discretization
    def PandasEqualWidthdiscretization(self):
        for (columnName, columnData) in self.dataset.iteritems():
            if issubdtype(self.dataset[columnName].dtype, np.number):
                self.dataset[columnName] = pd.cut(self.dataset[columnName], self.count_bins)

    # -------------------------- Our Equal width discretization
    def OurEqualWidthdiscretization(self):
        index = 0

        for (columnName, columnData) in self.dataset.iteritems():
            if issubdtype(self.dataset[columnName].dtype, np.number):
                minimum = min(self.dataset[columnName])
                maximum = max(self.dataset[columnName])
                width = round((maximum - minimum) / self.count_bins, 3)
                start, end = minimum, width + minimum

                dic = {}
                new_set = list(dict.fromkeys(self.dataset[columnName]))

                for k in range(self.count_bins):                                    # building the bins in equal width method
                    for i in new_set:
                        if type(i) is not pd.Interval and i >= start and i <= end:
                            dic[i] = pd.Interval(left=start, right=end)
                        index += 1
                    index = 0
                    start = end
                    end += width
                self.dataset[columnName] = self.dataset[columnName].map(dic)
    # -------------------------- Our Entropy
    def Entropy(self, labels):                      # regular entropy calculation
        n_labels = len(labels)
        if n_labels <= 1:                   # entropy of maximum 1 element
            return 0

        value, counts = np.unique(labels, return_counts=True)
        p = counts / n_labels                    # the ratio of each different element exist divided by total column len
        n_classes = np.count_nonzero(p)          # check how much elements are different from zero

        if n_classes <= 1:                       # if the column include only one element that repeat N times
            return 0
        ent = 0

        for i in p:                             # if the column is normal we will calculate his entropy
            ent -= i * log(i, 2)

        return ent

    def ConditionalEntropy(self, classification, column):
        def indices(lbl, attribute):
            return [i for i, j in enumerate(attribute) if j == lbl]             # return list of indexes (i) where lbl exist in column, j = each value in column

        cond_entropy = 0
        total = len(classification)
        for label in Counter(column).keys():                                # label = column values - keys
            sv = [classification[i] for i in indices(label, column)]        # sv = classification values of each label in column
            entropy = self.Entropy(sv)
            cond_entropy += entropy * len(sv) / total
        return cond_entropy

    def OurEntropy(self):
        
        class_entropy = self.Entropy(self.dataset['class'])
        bin_number, combination = self.count_bins, None

        for (columnName, columnData) in self.dataset.iteritems():
            if issubdtype(self.dataset[columnName].dtype, np.number):
                all_bins, new_column, best_col = [], [], []
                best_val, temp_val = -1000, 0
                order = self.dataset[columnName].tolist()
                order = list(dict.fromkeys(order))
                order.sort()
                combination = itertools.combinations(order, bin_number - 1)

                for i in combination:                       # checking all the possible options of splitting the bins
                    all_bins = []
                    len_i = len(i)
                    for k in i:                             # building the bind according to the combinations values
                        if len_i == 1:
                            if k == order[-1]:
                                continue
                            if k == order[0]:
                                all_bins.append(order[:1])
                                all_bins.append(order[1:])
                            else:
                                all_bins.append(order[:order.index(k) + 1])
                                all_bins.append(order[order.index(k) + 1:])
                        else:
                            if order[0] == k and k == i[0]:
                                all_bins.append([order[0]])
                            elif order[0] < k and k == i[0]:
                                all_bins.append(order[0:order.index(k) + 1])

                            elif i[0] < k < i[-1]:
                                x = i[i.index(k) - 1]
                                all_bins.append(order[order.index(x) + 1:order.index(k) + 1])

                            elif order[-1] > k and k == i[-1]:
                                x = i[i.index(k) - 1]
                                temp_order_kindex = order.index(k)
                                all_bins.append(order[order.index(x) + 1:temp_order_kindex + 1])
                                all_bins.append(order[temp_order_kindex + 1:])
                            elif order[-1] == k and k == i[-1]:
                                x = i[i.index(k) - 1]
                                all_bins.append(order[order.index(x) + 1:order.index(k)])
                                all_bins.append([order[-1]])

                    if all_bins:
                        new_column = []
                        for t in self.dataset[columnName]:
                            for l in range(len(all_bins)):
                                if t in all_bins[l]:
                                    new_column.append(pd.Interval(all_bins[l][0], all_bins[l][-1]))         #classified each value in the dataset column to the appropriate bin above
                                    break
                        temp_conditional = self.ConditionalEntropy(self.dataset['class'], new_column)
                        temp_val = class_entropy - temp_conditional                 #checking for the information gain

                        if temp_val > best_val and temp_conditional > 0 and temp_val > 0:
                            best_val = temp_val
                            best_col = new_column

                self.dataset[columnName] = best_col                 # update the current dataset to the new column we build by the bins with the best gain entropy
                # print(self.dataset[columnName])
                




