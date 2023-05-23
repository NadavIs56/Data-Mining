# נדב ישי - 20611989

from collections import Counter
from math import log
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
from Discretization import Discretization

class AlgoNaiveBayes:
    def __init__(self, datasets):
        self.datasets = datasets

    # -------------------------- Sklearn naive bayes
    def SklearnNaiveBayes(self):
        le = preprocessing.LabelEncoder()

        x_train = self.datasets[0].iloc[:, :-1]                     # Get all columns without a classification column fil train
        y_train = le.fit_transform(self.datasets[0].iloc[:, -1])               # Get the classification column only

        x_train = list(zip(*[le.fit_transform(columnData) for (columnName, columnData) in x_train.iteritems()]))
        model = GaussianNB()
        model.fit(x_train, y_train)

        temp_picle = open("Sklearn_NB_model", "wb")
        pickle.dump(model, temp_picle)
        temp_picle.close()

        x_test = self.datasets[1].iloc[:, :-1]
        x_test = list(zip(*[le.fit_transform(columnData) for (columnName, columnData) in self.datasets[1].iloc[:, :-1].iteritems()]))
        y_test = le.fit_transform(self.datasets[1].iloc[:, -1])
        classifcation = GaussianNB().fit(x_train, y_train)
        y_pred = classifcation.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred) * 100

        return ["Sklearn naive bayes accuracy = " + str(accuracy) + "%", str(classification_report(y_test, y_pred)), "Confusion_matrix: \n" + str(confusion_matrix(y_test, y_pred))]

    # -------------------------- our naive bayes
    def Our_NaiveBayes(self):

        def ReadFields(line):
            start = line.find("{") + len("{")
            end = line.find("}")
            substring = line[start:end]
            return list(substring.split(','))
        
        columns_names, total_columns, classes, temp = [], [], [], []

        train = self.datasets[0]
        test = self.datasets[2]
        train_classes = train['class']
        test_classes = test['class']
        number = LabelEncoder()
        for (columnName, columnData) in self.datasets[0].iteritems():              # scaling our data
            if type(self.datasets[0][columnName][1]) == 'str':
                self.datasets[0][columnName] = number.fit_transform(self.datasets[0][columnName])
                self.datasets[2][columnName] = number.fit_transform(self.datasets[2][columnName])

        class_yes = (train_classes == 'yes').sum()
        class_no = (train_classes == 'no').sum()

        train_without_class = self.datasets[0].loc[:, self.datasets[0].columns != 'class']
        test_without_class = self.datasets[2].loc[:, self.datasets[2].columns != 'class']

        columns_names = list(train_without_class.columns)
        count_dic = {}

        numeric_location = []
        index = 0
        for i in self.datasets[1]:
            if index < len(columns_names):
                count_dic[columns_names[index]] = {}
                if 'NUMERIC' in i:                              # create dictionary, each key is attribute, and the values are the values of each attribute
                    total_columns.append(0)
                    numeric_location.append(index)
                    lst = dict.fromkeys(train_without_class[columns_names[index]].tolist())
                    lst.update(dict.fromkeys(test_without_class[columns_names[index]].tolist()))
                    lst = list(dict.fromkeys(lst))
                else:
                    lst = ReadFields(i)
                    total_columns.append(lst)
                for j in lst:
                    count_dic[columns_names[index]][j] = [1, 1]
                index += 1

        for columnName, columnData in train_without_class.iteritems():
            lst = list(dict.fromkeys(train_without_class[columnName]))                  # create dictionary of chances - 80% of train file
            for j in lst:
                ychance = (train[(train["class"] == "yes") & (train[columnName] == j)].count())[0] / class_yes
                nchance = (train[(train["class"] == "no") & (train[columnName] == j)].count())[0] / class_no
                count_dic[columnName][j] = [ychance, nchance]

        temp_pickle = open("our_NB_model", "wb")            # saving the chances dictionary - model, by pickle
        pickle.dump(count_dic, temp_pickle)
        temp_pickle.close()

        test_without_class = self.datasets[2].loc[:, self.datasets[2].columns != 'class']
        columns_names = list(test_without_class)
        iterator = 0
        yes, no = 1, 1
        yes_no_list = []

        for i, row in test_without_class.iterrows():                    # check the chances of each row in test file by multiply all the chances from our dictionary + laplace correction
            for j in row:
                if iterator in numeric_location:
                    for k in count_dic[columns_names[iterator]].keys():
                        if k.left <= j <= k.right:
                            yes *= count_dic[columns_names[iterator]][k][0] + 1
                            no *= count_dic[columns_names[iterator]][k][1] + 1
                            break
                else:
                    yes *= count_dic[columns_names[iterator]][j][0] + 1
                    no *= count_dic[columns_names[iterator]][j][1] + 1

                iterator += 1

            if yes >= no:
                yes_no_list.append("yes")
            else:
                yes_no_list.append("no")
            yes = 1
            no = 1
            iterator = 0

        true_pos, false_pos = 0, 0
        true_neg, false_neg = 0, 0
        classes = self.datasets[2]['class']
        yes, no = 0, 0

        for i in range(len(classes)):                   # check my answers
            if yes_no_list[i] == classes[i]:
                yes += 1
                if yes_no_list[i] == "yes":
                    true_pos += 1
                else:
                    true_neg += 1
            else:
                no += 1
                if yes_no_list[i] == "yes":
                    false_pos += 1
                else:
                    false_neg += 1

        accuracy = yes / len(classes) * 100                                         # calculate the results: precision, recall, f1 and accuracy by myself - without Sklearn function
        yprecision, yrecall, yf1, nprecision, nrecall, nf1 = 0, 0, 0, 0, 0, 0

        if true_pos + false_pos != 0:
            yprecision = round(true_pos / (true_pos + false_pos), 2)
        if true_pos + false_neg != 0:
            yrecall = round(true_pos / (true_pos + false_neg), 2)
        if yprecision + yrecall != 0:
            yf1 = round(2 * (yprecision * yrecall) / (yprecision + yrecall), 2)

        if true_neg + false_pos != 0:
            nprecision = round(true_neg / (true_neg + false_pos), 2)
        if true_neg + false_neg != 0:
            nrecall = round(true_neg / (true_neg + false_neg), 2)
        if nprecision + nrecall != 0:
            nf1 = round(2 * (nprecision * nrecall) / (nprecision + nrecall), 2)

        our_report = {"precision": [yprecision, nprecision], "recall": [yrecall, nrecall], "f1-score": [yf1, nf1]}

        return ["Our naive bayes accuracy = " + str(accuracy) + "%\n", str(our_report), "Confusion_matrix: \n" + str([[true_pos, false_pos], [false_neg, true_neg]])]


# -------------------------- Algorithm decision
class AlgoDecisionTree:
    def __init__(self, datasets):
        self.datasets = datasets

    def SklearnDecisionTree(self):
        number = LabelEncoder()
        for i in self.datasets[0].columns:
            self.datasets[0][i] = number.fit_transform(self.datasets[0][i])

        number = LabelEncoder()
        for i in self.datasets[1].columns:
            self.datasets[1][i] = number.fit_transform(self.datasets[1][i])

        features = [i for i in self.datasets[0].columns[:-1]]
        target = "class"

        features_train, target_train = self.datasets[0][features], self.datasets[0][target]
        features_test, target_test = self.datasets[1][features], self.datasets[1][target]

        model = DecisionTreeClassifier(criterion='entropy', max_depth=9)                  # entropy gets better accuracy than gini
        model.fit(features_train, target_train)

        temp_picle = open("Sklearn_ID3_model", "wb")
        pickle.dump(model, temp_picle)
        temp_picle.close()

        pred = model.predict(features_test)
        accuracy = accuracy_score(target_test, pred) * 100

        return ["Sklearn naive bayes accuracy = " + str(accuracy) + "%", str(classification_report(target_test, pred)),
                "Confusion_matrix: \n" + str(confusion_matrix(target_test, pred))]


class TreeStruct:
    def __init__(self, parent):             # struct of tree for the ID3
        self.parent = parent
        self.node = {}

    def AddChild(self, node, children):
        self.node[node] = children

    def GetChild(self, node):
        if isinstance(node, str):
            return self.node[node]
        else:
            for i in self.node:
                if i.left <= node <= i.right:
                    return self.node[i]

    def GetParent(self):
        return self.parent

class ID3:
    def __init__(self, datasets):
        self.train = datasets[0]
        self.test = datasets[1]
        self.structure = datasets[2]
        self.total_bin = {}
        self.CreateBins()
        self.classEntropy = Discretization(self.train).Entropy(datasets[0]['class'])
        self.RemoveUnwantedColumns()

    def InformationGain(self, classification, col):             # using entropy and conditional entropy from Discretization file, in order not to create duplicates in the code
        return Discretization(self.train).Entropy(classification) - Discretization(self.train).ConditionalEntropy(classification, col)

    def RemoveUnwantedColumns(self):                # remove columns with information gain that lower than 90% from the highest information gain
        gain_list = []
        classification = self.train['class']

        for columnName, columnData in self.train.iteritems():
            if columnName != 'class':
                gain_list.append(self.classEntropy - Discretization(self.train).ConditionalEntropy(classification, self.train[columnName]))
        columns = self.train.columns
        best_gain = max(gain_list)
        index = 0
        for i in gain_list:
            if i < best_gain * 0.1:                                         # check the information gain and drop the column if necessary
                self.train = self.train.drop(columns=[columns[index]])
                self.test = self.test.drop(columns=[columns[index]])
            index += 1


    def GetColRatio(self, dataset):                         # find and return the best information gain column
        temp_gain, best_gain, col_name = 0, 0, ''
        for columnName, columnData in dataset.iteritems():
            if columnName != "class":
                temp_gain = self.InformationGain(dataset['class'], dataset[columnName])
                if temp_gain > best_gain:
                    best_gain = temp_gain
                    col_name = columnName

        return best_gain, col_name

    def FindMost(self, dataset):                            # find and return the most common classification word: yes / no
        yes = (dataset['class'] == 'yes').sum()
        no = (dataset['class'] == 'no').sum()

        if yes >= no:
            return 'yes'
        return 'no'

    def BuildID3(self, dataset, total_bin, depth):

        if depth == 0:                          # threshold according to the tree's depth
            return self.FindMost(dataset)

        val, node_name = self.GetColRatio(dataset)

        if node_name == '' or val >= 1:         # threshold according to information gain bad value or no more columns
            return self.FindMost(dataset)

        t = TreeStruct(node_name)                   # recursion that build the tree
        for i in self.total_bin[node_name]:
            t.AddChild(i, self.BuildID3(dataset.loc[dataset[node_name] == i].drop(columns=[node_name]).reset_index(drop=True), total_bin, depth - 1))
        return t

    def Test(self):
        yes, no = 0, 0
        columns_number = len(self.train.columns) - 1
        depth_list = []          # tree depth that I found out to be the most efficient according to a lot of executions
        if columns_number == 3:
            depth_list.append(2)
            depth_list.append(3)
        elif columns_number == 4:
            depth_list.append(3)
            depth_list.append(4)
        elif columns_number > 4:
            depth_list.append(3)
            depth_list.append(4)
            depth_list.append(int((len(self.train.columns) - 1) * 0.5))
            depth_list.append(int((len(self.train.columns) - 1) * 0.667))

        depth_list = list(dict.fromkeys(depth_list))        # remove duplicates in order to check each value only once
        ctp, ctn, cfp, cfn = 0, 0, 0, 0
        accuracy, depth = 0, 0
        copy_model = 0

        for j in depth_list:
            copy_train = self.train.copy()                                  # checking some different depth I found to be with the best accuracy
            train_result = self.BuildID3(copy_train, self.total_bin, j)
            tp, tn, fp, fn = 0, 0, 0, 0

            for i, row in self.test.iterrows():
                copy = train_result
                while (isinstance(copy, TreeStruct)):
                    copy = copy.GetChild(row[copy.GetParent()])
                if row['class'] == copy:
                    if copy == "yes":
                        tp += 1
                    else:
                        tn += 1
                    yes += 1
                else:
                    if copy == "yes":
                        fp += 1
                    else:
                        fn += 1
                    no += 1
            if (yes / (yes + no)) * 100 > accuracy:
                accuracy = (yes / (yes + no)) * 100
                ctp, ctn, cfp, cfn = tp, tn, fp, fn
                depth = j
                copy_model = train_result

        temp_picle = open("Our_ID3_model", "wb")            # saving the model using pickle
        pickle.dump(copy_model, temp_picle)
        temp_picle.close()

        yprecision, yrecall, yf1, nprecision, nrecall, nf1 = 0, 0, 0, 0, 0, 0   # results analyzing, and making report

        if ctp + cfp != 0:
            yprecision = round(ctp / (ctp + cfp), 2)
        if ctp + cfn != 0:
            yrecall = round(ctp / (ctp + cfn), 2)                                       # calculate the results: precision, recall, f1 and accuracy by myself - without Sklearn function
        if yprecision + yrecall != 0:
            yf1 = round(2 * (yprecision * yrecall) / (yprecision + yrecall), 2)

        if ctn + cfp != 0:
            nprecision = round(ctn / (ctn + cfp), 2)
        if ctn + cfn != 0:
            nrecall = round(ctn / (ctn + cfn), 2)
        if nprecision + nrecall != 0:
            nf1 = round(2 * (nprecision * nrecall) / (nprecision + nrecall), 2)

        our_report = {"precision": [yprecision, nprecision], "recall": [yrecall, nrecall], "f1-score": [yf1, nf1]}

        return ["Our decision tree accuracy = " + str(accuracy) + "%\n", "Depth limit = " + str(depth) + "\n", "Number of columns after removing the columns with bad information gain = "+ str(columns_number) + "\n",
                str(our_report), "Confusion_matrix: \n" + str([[ctp, cfp], [cfn, ctn]])]


    def CreateBins(self):
        columns_names = list(self.train.loc[:, self.train.columns != 'class'].columns)
        train_without_class = self.train.loc[:, self.train.columns != 'class']
        count_dic = {}
        index = 0
        for i in self.structure:
            if index < len(columns_names):
                count_dic[columns_names[index]] = {}
                if 'NUMERIC' in i:
                    lst = list(dict.fromkeys(train_without_class[columns_names[index]].tolist()))
                else:
                    start = i.find("{") + len("{")
                    end = i.find("}")
                    substring = i[start:end]
                    lst = list(substring.split(','))
                count_dic[columns_names[index]] = lst
                index += 1
        self.total_bin = count_dic


# -------------------------- Sklearn KNN
class KNN:
    def __init__(self, datasets):
        self.datasets = datasets
    def SklearnKNN(self):
        number = LabelEncoder()
        for i in self.datasets[0].columns:
            self.datasets[0][i] = number.fit_transform(self.datasets[0][i])

        number = LabelEncoder()
        for i in self.datasets[1].columns:
            self.datasets[1][i] = number.fit_transform(self.datasets[1][i])

        features = [i for i in self.datasets[0].columns[:-1]]
        target = "class"

        features_train, target_train = self.datasets[0][features], self.datasets[0][target]
        features_test, target_test = self.datasets[1][features], self.datasets[1][target]
        error = []
        index = 0
        temp_ac, ac, save_pred, save_acc = 0, 0, 0, 0

        for i in range(1, 11):                              # Calculating error for K values between 1 and 11
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(features_train, target_train)
            pred_i = knn.predict(features_test)
            print(i)
            temp_ac = accuracy_score(target_test, pred_i)
            print(temp_ac * 100)
            if temp_ac > ac:
                ac = temp_ac
                index = i
                save_pred = pred_i
            error.append(np.mean(pred_i != target_test))
        print("winner:")
        print(ac)
        print(index)
        print(error)

        temp_picle = open("Sklearn_KNN_model", "wb")                        # saving the model using pickle
        pickle.dump(KNeighborsClassifier(n_neighbors=index), temp_picle)
        temp_picle.close()

        plt.figure(figsize=(12, 6))                                                     # create graph in order to see the results better and analyze them
        plt.plot(range(1, 11), error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('Error Rate K Value\nThe minimum error accepted by k = ' + str(index) + '\n' + 'Accuracy of: ' + str(round(ac * 100, 3)) + '%, Mean Error value = ' + str(error[index - 1]))
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')

        plt.savefig("KNN graph")
        plt.show()

        return ["Sklearn KNN accuracy = " + str(ac * 100) + "%", str(classification_report(target_test, save_pred)),
                "Confusion_matrix: \n" + str(confusion_matrix(target_test, save_pred))]

# -------------------------- Sklearn KMEANS
class Kmeans:
    def __init__(self, datasets):
        self.datasets = datasets

    def SklearnKMeans(self):
        number = LabelEncoder()
        for i in self.datasets[0].columns:
            self.datasets[0][i] = number.fit_transform(self.datasets[0][i])
        for i in self.datasets[1].columns:
            self.datasets[1][i] = number.fit_transform(self.datasets[1][i])

        features = [i for i in self.datasets[0].columns[:-1]]
        target = "class"
        features_train, target_train = self.datasets[0][features], self.datasets[0][target]
        features_test, target_test = self.datasets[1][features], self.datasets[1][target]

        mms = MinMaxScaler()                    # scale with min-max to give equal importance
        mms.fit(features_train)
        data_transformed = mms.transform(features_train)

        index, best_acc, accuracy, pred, save_pred = 0, 0, 0, 0, 0
        Sum_of_squared_distances, error = [], []
        K = range(1, 11)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(data_transformed)
            Sum_of_squared_distances.append(km.inertia_)
            pred = km.predict(features_test)
            error.append(np.mean(pred != target_test))
            accuracy = accuracy_score(target_test, pred)
            if accuracy > best_acc:
                best_acc = accuracy
                index = k
                save_pred = pred
            print("k = " + str(k))
            print("accuracy = " + str(accuracy * 100))
            print("sum of squared distances = " + str(Sum_of_squared_distances[k-1]))
        print("The best k = " + str(index) + " Accuracy = " + str(best_acc * 100))
        print("Sum of squared distances = " + str(Sum_of_squared_distances[index - 1]))
        print("Mean error value = " + str(error[index - 1]))

        temp_picle = open("Sklearn_K-Means_model", "wb")            # saving the model using pickle
        pickle.dump(KMeans(n_clusters=index), temp_picle)
        temp_picle.close()

        plt.figure(figsize=(15, 8))

        plt.subplot(1, 2, 1)                                    # show 2 graphs for better analyzing and understanding
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k Value')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k\nThe optimal k = ' + str(index) + ", Accuracy = " + str(round(best_acc * 100, 3)) + "%\n" + "Sum_of_squared_distances for k = " + str(index) + " is: " + str(Sum_of_squared_distances[index - 1]))

        plt.subplot(1, 2, 2)
        plt.plot(K, error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
        plt.title('Error Rate K Value\n' + "The minimum mean error = " + str(round(error[index-1], 3)) + ", obtained by k = " + str(index))
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')

        plt.savefig("K-Means graph")
        plt.show()

        return ["Sklearn K-Means accuracy = " + str(round(best_acc * 100, 3)) + "%", str(classification_report(target_test, save_pred)),
                "Confusion_matrix: \n" + str(confusion_matrix(target_test, save_pred))]