import streamlit as st
import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import *
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

training = pd.read_csv("C:/Users/P MAHENDRA/OneDrive/Desktop/csp/Training.csv")
testing = pd.read_csv("C:/Users/P MAHENDRA/OneDrive/Desktop/csp/Testing.csv")
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

#print(accuracy_score(clf.predict(x_train),y_train))
#print(accuracy_score(clf.predict(x_test),y_test))


scores = cross_val_score(clf, x_test, y_test, cv=3)
#st.write("Decision Tree Classifier Mean Accuracy:", scores.mean())

model = SVC()
model.fit(x_train, y_train)
#st.write("SVM Classifier Score:", model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        if item in severityDictionary:
            sum = sum + severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        st.write("You should take consultation from a doctor.")
    else:
        st.write("It might not be that bad, but you should take precautions.")


def getDescription():
    global description_list
    with open("symptom_Description.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open("Symptom_severity.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open("symptom_precaution.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp):
    df = pd.read_csv("Training.csv")
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []


    disease_input = st.selectbox("Enter Your Symptom", cols)
       

    while True:
        try:
            num_days = st.number_input("Okay. From how many days? : ",min_value=1)
            break
        except:
            print("Enter valid input.")

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            st.write("Are you experiencing any ")
            symptoms_exp = []
            for syms in symptoms_given:
                inp = st.selectbox(f"Are you experiencing {syms}?", ["Yes", "No"])
                if inp == "Yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                st.header("You may have ", present_disease[0])
                st.write(description_list[present_disease[0]])
            else:
                st.write("You may have ", present_disease[0], "or ", second_prediction[0])
                st.write(description_list[present_disease[0]])
                st.write(description_list[second_prediction[0]])

            precution_list = precautionDictionary[present_disease[0]]
            st.write("Take following measures:")
            for i, j in enumerate(precution_list):
                st.write(i + 1, ")", j)

    recurse(0, 1)


st.title("Prana -(A Disease Predictor)")
st.sidebar.header("User Information")
name = st.sidebar.text_input("Your Name")


getSeverityDict()
getDescription()
getprecautionDict()
tree_to_code(clf, cols)

