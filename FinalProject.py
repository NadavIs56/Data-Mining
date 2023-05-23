# נדב ישי - 20611989

import tkinter as tk
from tkinter import *
from tkinter import ttk, messagebox
import os
import pandas as pd

#Import for all our different classes for the project models
from Discretization import Discretization
from Normalization import Normalization
from Completion import Completion
from Classifications import *
from datetime import datetime

#---------------------------frame
root = tk.Tk()
root.title("Final Project")
root.resizable(False, False)
canvas = tk.Canvas(root, height=600, width=1000, bg="#263D42")
canvas.pack()
frame = tk.Frame(root, bg="white")
frame.place(relwidth=0.9, relheight=0.9, relx=0.05, rely=0.05)

#-------------------------- open files
openFilesLabel = Label(frame, text="Click to load train & test & Structure files:", bg="white")
openFilesLabel.place(x=100, y=70)
loadFiles = tk.Button(frame, text="Load Files", padx=15, pady=8, fg="white", bg="#3071a9", command=lambda: ReadFiles())
loadFiles.place(x=330, y=66, height=30, width=100)
path_var = StringVar()
files_path = Entry(frame, textvariable=path_var, width=60)
files_path.place(x=490, y=73)
noteLabel = Label(frame, text="Note, if the path box is empty, the files will be loaded from the main project folder by defult.", bg="white", fg='purple')
noteLabel.place(x=100, y=35)
pathLabel = Label(frame, text="Path:", bg="white")
pathLabel.place(x=450, y=73)
loadLabel1 = Label(frame, text="First, please load your files!", bg="white", fg='purple')
loadLabel1.place(x=100, y=10)
loadLabel2 = Label(frame, text="Files loaded successfully.", bg="white", fg='#0099cc')

#-------------------------- missing values compeltion
missing_val_var = IntVar()
missing_val_var.set(1)
valuesCompletionLabel = Label(frame, text="Choose missing values completion type:", bg="white")
valuesCompletionLabel.place(x=100, y=120)
R1 = Radiobutton(frame, text="By classification column", variable=missing_val_var, value=1,  activebackground='white')
R1.place(x=335, y=120)
R2 = Radiobutton(frame, text="By current dataset", variable=missing_val_var, value=2,  activebackground='white')
R2.place(x=495, y=120)

#-------------------------- nirmalization
normalization_var = tk.StringVar()
normalization_combobox = ttk.Combobox(frame, width=35, textvariable=normalization_var, values=['None', 'Z Score', 'Decimal Scaling', 'Min-Max'])
normalization_combobox.place(x=260, y=175)
normalization_combobox.set("None")
normalizationLabel = Label(frame, text="Choose normalization type:", bg="white")
normalizationLabel.place(x=100, y=175)

#-------------------------- discretization
discretization_var = tk.StringVar()
discretization_combobox = ttk.Combobox(frame, width=35, textvariable=discretization_var, values=['None', 'Pandas equal frequency discretization', 'Our equal frequency discretization', 'Pandas equal width discretization', 'Our equal width discretization', 'Our entropy based discretization'])
discretization_combobox.place(x=260, y=225)
discretization_combobox.set("None")
discritizationLabel = Label(frame, text="Choose discritization type:", bg="white")
discritizationLabel.place(x=100, y=225)

disLabel1 = Label(frame, text="Please choose discretization while selecting this model.", bg="white", fg='red')
disLabel2 = Label(frame, text="Please don't choose discretization while selecting this model.", bg="white", fg='red')

binNunber_var = tk.IntVar()
binNumber_list = []
binNumberLabel = Label(frame, text="Choose number of bins:", bg="white")
binNumber_combobox = ttk.Combobox(frame, width=35, textvariable=binNunber_var, values=binNumber_list)

#-------------------------- algoorithm choose
algoDecision_var = tk.StringVar()

algoDecision_combobox = ttk.Combobox(frame, width=35, textvariable=algoDecision_var, values=['None', 'Sklearn naive bayes algorithm', 'Our naive bayes algorithm', 'Sklearn decision tree algorithm', 'Our decision tree algorithm', 'Sklearn KNN', 'Sklearn KMeans'])
algoDecision_combobox.place(x=260, y=330)
algoDecision_combobox.set("None")
algoDecisionLabel = Label(frame, text="Choose model's algorithm:", bg="white")
algoDecisionLabel.place(x=100, y=330)

kmeansLabel = Label(frame, text="For better results, K-Means using Min-Max normalization by default.", bg="white", fg='green')
# ----- refresh bins combobox after changing discretization type
def RefreshBins(event):
    global discretization_var, binNumberLabel, binNumber_combobox
    discretization_choice = event.widget.get()

    if discretization_choice != "None":
        binNumberLabel.place(x=100, y=280)
        binNumber_combobox.place(x=260, y=280)
        binNumber_combobox['state'] = 'NORMAL'

    else:
        binNumber_combobox['state'] = 'disabled'

discretization_combobox.bind("<<ComboboxSelected>>", RefreshBins, True)

#-------------------------- discretization

submit = tk.Button(frame, text="Submit", padx=0, pady=5, fg="white", bg="#1c2730", command=lambda: Calculate(), state='disabled')
submit.place(x=370, y=480, height=50, width=150)

test_file, train_file, structure_file, save_file, start_time, flag = None, None, None, open("running_results.txt", 'w'), 0, True


#-------------------------- read files function
def ReadFiles():
    global test_file, train_file, submit, binNunber_var, binNumber_combobox, structure_file, flag, loadLabel1, loadFiles, path_var
    path = str(path_var.get())
    path += "\\"
                                                            # load the structure, train and test files from path, or from the main project folder by defult
    try:
        if path_var.get() != "":
            structure_file = open(path + "Structure.txt", "r")
        else:
            structure_file = open("Structure.txt", "r")
        if os.stat("Structure.txt").st_size == 0:
            messagebox.showerror("Error", "Error: Structure file is empty")
            flag = False
    except TypeError or FileNotFoundError:
        messagebox.showerror("structure file missing", "Error: Structure file does not appear to exist.\nPlease include structure.txt file in the selected folder")
        flag = False
        return
    try:
        if path_var.get() != "":
            test_file = pd.read_csv(path + "test.csv")
        else:
            test_file = pd.read_csv("test.csv")
        if test_file.empty:
            messagebox.showerror("Error", "Error: Test file is empty")
            flag = False
        else:
            test_file = test_file.dropna(subset=["class"])  # test file remove  all rows with class None
    except TypeError or Exception:
        messagebox.showerror("test file missing", "Error: Test file does not appear to exist.\nPlease include test.csv file in the selected folder")
        flag = False
        return
    try:
        if path_var.get() != "":
            train_file = pd.read_csv(path + "train.csv")
        else:
            train_file = pd.read_csv("train.csv")
        if train_file.empty:
            messagebox.showerror("Error", "Error: Train file is empty")
            flag = False
        else:
            train_file = train_file.dropna(subset=["class"])  # train file - remove  all rows with class None
    except TypeError or Exception:
        messagebox.showerror("train file missing", "Error: Train file does not appear to exist.\nPlease include train.csv file in the selected folder")
        flag = False
        return
    if flag:
        messagebox.showinfo("All done", "Files loaded successfully")

        count = len(train_file.index)           # calculate the number of bins we want to enable to the user
        if count > 1000:
            count = 30
        else:
            count = int(count ** 0.5)
        for i in range(1, count + 1):
            binNumber_list.append(i + 1)
        binNumber_combobox['values'] = binNumber_list
        binNumber_combobox.current(0)

        loadLabel1.place_forget()
        noteLabel.place_forget()
        loadLabel2.place(x=100, y=25)
        loadFiles['state'] = 'disabled'
        flag = False

#-------------------------- Enable submit button only if discretization was selected while choosing algorithm model
def EnableSubmit(event):
    global discretization_var, algoDecision_var, submit, flag, normalization_combobox, kmeansLabel, disLabel1, disLabel2
    discretization_choice = discretization_var.get()
    algo_choice = algoDecision_var.get()
                                                                    # function that check if the discretization matches the model type
    if algo_choice == 'Sklearn KMeans':                             # checking for illegal combinations
        kmeansLabel.place(x=100, y=360)
        normalization_combobox.current(0)
        normalization_combobox["state"] = 'disabled'
    else:
        kmeansLabel.place_forget()
        normalization_combobox["state"] = 'normal'
    if flag:
        submit["state"] = 'disabled'
    elif discretization_choice == "None" and (algo_choice != 'Sklearn KNN' and algo_choice != 'Sklearn KMeans'):
        submit["state"] = 'disabled'
        disLabel2.place_forget()
        disLabel1.place(x=100, y=250)
    elif discretization_choice != "None" and (algo_choice == 'Sklearn KNN' or algo_choice == 'Sklearn KMeans'):
        submit["state"] = 'disabled'
        disLabel1.place_forget()
        disLabel2.place(x=100, y=250)
    else:
        submit["state"] = 'normal'
        disLabel1.place_forget()
        disLabel2.place_forget()

    if discretization_choice == "None" and algo_choice == 'None':
        disLabel1.place_forget()
        disLabel2.place_forget()
        submit["state"] = 'disabled'
discretization_combobox.bind("<<ComboboxSelected>>", EnableSubmit, True)
algoDecision_combobox.bind("<<ComboboxSelected>>", EnableSubmit, True)


#-------------------------- Calculate for the submit button action = command
def Calculate():                                       # run the function by specific order: normal', disc', saving clean files and run model
    global train_file, start_time
    start_time = datetime.now()
    execute_values_completion()
    execute_normalization()
    execute_discretization()
    clean()
    execute_algorithm()
    save_file.close()
    osCommandString = "notepad.exe running_results.txt"
    os.system(osCommandString)
    exit()
#-------------------------- Manage the values completion
def execute_values_completion():
    global missing_val_var, save_file
    compl = Completion([train_file, test_file])

    if missing_val_var.get() == 1:
        save_file.write("Missing values completed by classification column.\n\n")
        compl.CompleteByClassification()
    else:
        save_file.write("Missing values completed by current dataset.\n\n")
        compl.CompleteByCurrentDataSet()


#-------------------------- manage all the normalization methods
def execute_normalization():
    global normalization_var, save_file
    normalization_choice = normalization_var.get()
    normal = Normalization(train_file)

    save_file.write("Normalization type = " + normalization_choice + ".\n\n")

    if normalization_choice == "Z Score":
        normal.ZScore()
    elif normalization_choice == "Min-Max":
        normal.MinMax()
    elif normalization_choice == "Decimal Scaling":
        normal.DecimalScaling()
        
#-------------------------- manage all the discretization methods
def execute_discretization():
    global discretization_var, save_file
    disc = Discretization(train_file, binNunber_var.get())

    if discretization_var.get() == None:
        save_file.write("Discretization type = " + discretization_var.get() + ", number of bins: " + str(0) + ".\n\n")
    else:
        save_file.write("Discretization type = " + discretization_var.get() + ", number of bins: " + str(binNunber_var.get()) + ".\n\n")

    if discretization_var.get() == "Pandas equal frequency discretization":
        disc.PandasEqualFrequencydiscretization()
    elif discretization_var.get() == "Pandas equal width discretization":
        disc.PandasEqualWidthdiscretization()
    elif discretization_var.get() == "Our equal frequency discretization":
       disc.OurEqualFrequencydiscretization() 
    elif discretization_var.get() == "Our equal width discretization":
        disc.OurEqualWidthdiscretization()
    elif discretization_var.get() == "Our entropy based discretization":
        disc.OurEntropy()


# -------------------------- Saving the clean files
def clean():
    global train_file, test_file

    clean_train_file = "train_clean.csv"
    train_file.to_csv(clean_train_file, encoding='utf-8', index=False)


    clean_test_file = "test_clean.csv"
    test_file.to_csv(clean_test_file, encoding='utf-8', index=False)

# -------------------------- Algorithms
def execute_algorithm():
    global algoDecision_var, start_time
    results = []
    algo = algoDecision_var.get()

    if algo == "Our naive bayes algorithm":
        results = AlgoNaiveBayes([train_file, structure_file, test_file]).Our_NaiveBayes()
    elif algo == "Sklearn naive bayes algorithm":
        results = AlgoNaiveBayes([train_file, test_file]).SklearnNaiveBayes()
    elif algo == "Sklearn decision tree algorithm":
        results = AlgoDecisionTree([train_file, test_file]).SklearnDecisionTree()
    elif algo == "Our decision tree algorithm":
        results = ID3([train_file, test_file, structure_file]).Test()
    elif algo == "Sklearn KNN":
        results = KNN([train_file, test_file]).SklearnKNN()
    elif algo == "Sklearn KMeans":
        results = Kmeans([train_file, test_file]).SklearnKMeans()

    if algoDecision_var.get() != "None":                              # export all the execution data to "running_result" file
        save_file.write("Model: " + algoDecision_var.get() + "\n\n")
        for i in results:
            save_file.write(i + "\n\n")
    else:
        save_file.write("Model was not selected\n\n\n")

    save_file.write("Total running time = " + str(datetime.now() - start_time))


mainloop()


