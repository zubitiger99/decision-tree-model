import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Load the model and encoders
@st.cache_resource
def load_model():
    with open('decision_tree_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

try:
    model, encoders = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Model files not found. Please ensure 'decision_tree_model.pkl' and 'label_encoders.pkl' are in the directory.")

# Title and description
st.title("🚢 Titanic Survival Prediction")
st.markdown("""
This application uses a **Decision Tree Classifier** to predict whether a passenger 
would have survived the Titanic disaster based on their characteristics.
""")

st.markdown("---")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Passenger Information")
    
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: f"Class {x} ({'Upper' if x==1 else 'Middle' if x==2 else 'Lower'})"
    )
    
    sex = st.radio("Gender", options=["male", "female"])
    
    age = st.slider("Age", min_value=0, max_value=80, value=30)
    
    fare = st.number_input("Fare (£)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

with col2:
    st.subheader("👨‍👩‍👧‍👦 Family Information")
    
    sibsp = st.number_input(
        "Number of Siblings/Spouses Aboard",
        min_value=0,
        max_value=8,
        value=0
    )
    
    parch = st.number_input(
        "Number of Parents/Children Aboard",
        min_value=0,
        max_value=6,
        value=0
    )
    
    embarked = st.selectbox(
        "Port of Embarkation",
        options=["C", "Q", "S"],
        format_func=lambda x: f"{x} - {'Cherbourg' if x=='C' else 'Queenstown' if x=='Q' else 'Southampton'}"
    )

st.markdown("---")

# Prediction button
if st.button("🔮 Predict Survival", type="primary", use_container_width=True):
    if model_loaded:
        # Prepare input data
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [encoders['sex'].transform([sex])[0]],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked': [encoders['embarked'].transform([embarked])[0]]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 1:
                st.success("### ✅ SURVIVED")
                st.markdown(f"**Confidence:** {prediction_proba[1]*100:.2f}%")
            else:
                st.error("### ❌ DID NOT SURVIVE")
                st.markdown(f"**Confidence:** {prediction_proba[0]*100:.2f}%")
        
        with result_col2:
            st.markdown("### Probability Distribution")
            prob_df = pd.DataFrame({
                'Outcome': ['Did Not Survive', 'Survived'],
                'Probability': [prediction_proba[0]*100, prediction_proba[1]*100]
            })
            st.bar_chart(prob_df.set_index('Outcome'))
        
        # Display input summary
        st.markdown("---")
        st.subheader("📝 Input Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Passenger Class", pclass)
            st.metric("Gender", sex.capitalize())
            st.metric("Age", f"{age} years")
        
        with summary_col2:
            st.metric("Fare", f"£{fare:.2f}")
            st.metric("Siblings/Spouses", sibsp)
            st.metric("Parents/Children", parch)
        
        with summary_col3:
            embark_name = "Cherbourg" if embarked == "C" else "Queenstown" if embarked == "Q" else "Southampton"
            st.metric("Embarkation Port", embark_name)
            st.metric("Total Family", sibsp + parch)

# Sidebar with additional information
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
**Model Information:**
- Algorithm: Decision Tree Classifier
- Dataset: Titanic - Kaggle
- Features: 7 input features
- Accuracy: ~80%

**Features Used:**
- Passenger Class
- Gender
- Age
- Number of Siblings/Spouses
- Number of Parents/Children
- Fare
- Port of Embarkation
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Model Performance")
st.sidebar.markdown("""
The model was trained on the famous Titanic dataset 
and achieves good accuracy in predicting survival outcomes.
""")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Decision Tree Classifier</p>
    </div>
    """,
    unsafe_allow_html=True
)