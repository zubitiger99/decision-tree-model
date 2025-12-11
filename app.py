import streamlit as st
import pandas as pd
import pickle

# Load SVM model
model = pickle.load(open("svm_model.pkl", "rb"))

st.title("Iris Flower Classifier (SVM Model)")
st.write("Enter the measurements below to predict Iris species.")

# User inputs
sepal_length = st.number_input("Sepal Length", 0.0, 10.0)
sepal_width = st.number_input("Sepal Width", 0.0, 10.0)
petal_length = st.number_input("Petal Length", 0.0, 10.0)
petal_width = st.number_input("Petal Width", 0.0, 10.0)

if st.button("Predict"):
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(data)[0]
    st.success(f"Predicted Species: {prediction}")
