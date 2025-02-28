import streamlit as st
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import csv

# Disable warnings for deprecated features
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Read the training and testing data
training = pd.read_csv("C:/Users/P MAHENDRA/OneDrive/Desktop/csp/Training.csv")
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']

# Encode labels
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Create a Decision Tree Classifier
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

# Initialize Streamlit app
st.title("Svastya -- Diease Prediction ChatBot")

# Input form for user's name (in sidebar)
st.sidebar.header("User Information")
name = st.sidebar.text_input("Your Name")
st.header("Hello!!",name)

# Input form for symptoms (as a selectbox)
st.header("Enter Your Symptoms:")
symptoms = st.multiselect("Select symptoms:", cols)

# Input form for number of days
num_days = st.number_input("Enter number of days you've had these symptoms:", min_value=1)

# Input form for additional symptoms (yes or no)
additional_symptoms = st.radio("Are you experiencing any additional symptoms which are not in above selectbox? ", ["Yes", "No"])

# Create a button to trigger prediction
if st.button("Predict"):
    # Perform prediction based on selected symptoms
    selected_symptoms = list(set(symptoms))  # Remove duplicates, if any
    input_vector = np.zeros(len(cols))
    for symptom in selected_symptoms:
        if symptom in cols:
            input_vector[cols.get_loc(symptom)] = 1

    disease_prediction = clf.predict([input_vector])[0]
    
    # Display disease prediction
    predicted_disease = le.inverse_transform([disease_prediction])[0]
    st.write(f"Based on the provided symptoms, you may have: {predicted_disease}")
    
    # Fetch description and precautions for the predicted disease (replace with your dataset)
    with open("D:/Desktop/disease/symptom_Description.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        description_dict = {row[0]: row[1] for row in csv_reader}
    
    with open("D:/Desktop/disease/symptom_precaution.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        precaution_dict = {row[0]: [row[1], row[2], row[3], row[4]] for row in csv_reader}
    
    if predicted_disease in description_dict:
        st.write("Description of the disease:")
        st.write(description_dict[predicted_disease])
    
    if predicted_disease in precaution_dict:
        st.write("Precautions:")
        st.write(precaution_dict[predicted_disease])
