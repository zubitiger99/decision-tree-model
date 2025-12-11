import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Page configuration
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open('decision_tree_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()
iris = load_iris()

# Title and description
st.title("🌸 Iris Flower Classification")
st.markdown("""
This app uses a **Decision Tree Classifier** to predict the species of Iris flowers
based on their measurements.
""")

# Sidebar for input
st.sidebar.header("Input Features")
st.sidebar.markdown("Adjust the sliders to input flower measurements:")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.8)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.2)
    
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Input Parameters")
    st.dataframe(input_df, use_container_width=True)

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

with col2:
    st.subheader("🎯 Prediction")
    species = iris.target_names[prediction[0]]
    
    if species == 'setosa':
        st.success(f"### {species.capitalize()}")
    elif species == 'versicolor':
        st.info(f"### {species.capitalize()}")
    else:
        st.warning(f"### {species.capitalize()}")
    
    st.subheader("📈 Prediction Probability")
    proba_df = pd.DataFrame(
        prediction_proba,
        columns=iris.target_names
    ).T
    proba_df.columns = ['Probability']
    st.bar_chart(proba_df)

# Model information
st.markdown("---")
st.subheader("ℹ️ About the Model")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algorithm", "Decision Tree")
with col2:
    st.metric("Dataset", "Iris")
with col3:
    st.metric("Features", "4")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Decision Tree Classifier")