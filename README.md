# **🚢 Titanic Survival Prediction \- Decision Tree Classifier**

A machine learning web application that predicts passenger survival on the Titanic using a Decision Tree Classifier, deployed with Streamlit.

## **📋 Project Overview**

This project uses the famous Titanic dataset from Kaggle to train a Decision Tree model that predicts whether a passenger would have survived the disaster based on various features like class, age, gender, and family size.

## **🚀 Features**

* **Interactive Web Interface**: User-friendly Streamlit dashboard  
* **Real-time Predictions**: Get instant survival predictions  
* **Probability Distribution**: View confidence scores for predictions  
* **Feature Importance Analysis**: Understand which factors matter most  
* **Model Persistence**: Pre-trained model saved for quick deployment

## **📊 Dataset**

**Source**: [Titanic \- Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/competitions/titanic/data)

**Features Used**:

* `Pclass`: Passenger Class (1, 2, 3\)  
* `Sex`: Gender (male/female)  
* `Age`: Age in years  
* `SibSp`: Number of siblings/spouses aboard  
* `Parch`: Number of parents/children aboard  
* `Fare`: Passenger fare  
* `Embarked`: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)

## **🛠️ Installation & Setup**

### **Prerequisites**

* Python 3.8 or higher  
* pip package manager

### **Step 1: Clone the Repository**

git clone https://github.com/yourusername/titanic-decision-tree.git  
cd titanic-decision-tree

### **Step 2: Install Dependencies**

pip install \-r requirements.txt

### **Step 3: Download Dataset**

1. Go to [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data)  
2. Download `train.csv`  
3. Place it in the project root directory

### **Step 4: Train the Model**

python train\_model.py

This will:

* Load and preprocess the data  
* Train the Decision Tree Classifier  
* Save the model as `decision_tree_model.pkl`  
* Save encoders as `label_encoders.pkl`  
* Display model performance metrics

### **Step 5: Run the Streamlit App**

streamlit run app.py

The app will open in your browser at `http://localhost:8501`

## **📁 Project Structure**

titanic-decision-tree/  
│  
├── app.py                          \# Streamlit web application  
├── train\_model.py                  \# Model training script  
├── decision\_tree\_model.pkl         \# Trained model (generated)  
├── label\_encoders.pkl              \# Label encoders (generated)  
├── requirements.txt                \# Python dependencies  
├── README.md                       \# Project documentation  
└── train.csv                       \# Titanic dataset (download separately)

## **🎯 Model Performance**

The Decision Tree Classifier achieves:

* **Accuracy**: \~80% on test data  
* **Precision**: High for both survival and non-survival classes  
* **Recall**: Balanced across classes

### **Feature Importance**

The most important features for prediction are typically:

1. Gender (Sex)  
2. Passenger Class (Pclass)  
3. Fare  
4. Age

## **🌐 Deployment on GitHub**

### **Option 1: Streamlit Cloud (Recommended)**

1. Push your code to GitHub:

git add .  
git commit \-m "Initial commit"  
git push origin main

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)  
3. Sign in with GitHub  
4. Click "New app"  
5. Select your repository  
6. Set main file path to `app.py`  
7. Click "Deploy"

### **Option 2: Heroku**

1. Create `Procfile`:

web: sh setup.sh && streamlit run app.py

2. Create `setup.sh`:

mkdir \-p \~/.streamlit/  
echo "\\  
\[server\]\\n\\  
headless \= true\\n\\  
port \= $PORT\\n\\  
enableCORS \= false\\n\\  
\\n\\  
" \> \~/.streamlit/config.toml

3. Deploy:

heroku create your-app-name  
git push heroku main

## **💻 Usage**

1. Open the web application  
2. Enter passenger information:  
   * Select passenger class  
   * Choose gender  
   * Set age using slider  
   * Enter fare amount  
   * Specify family members aboard  
   * Select embarkation port  
3. Click "Predict Survival"  
4. View the prediction results and probability distribution

## **🔧 Model Tuning**

To improve model performance, you can adjust hyperparameters in `train_model.py`:

dt\_model \= DecisionTreeClassifier(  
    max\_depth=5,              \# Maximum tree depth  
    min\_samples\_split=20,      \# Minimum samples to split  
    min\_samples\_leaf=10,       \# Minimum samples in leaf  
    random\_state=42  
)

## **📈 Future Enhancements**

* \[ \] Add more ML algorithms (Random Forest, SVM, XGBoost)  
* \[ \] Implement model comparison dashboard  
* \[ \] Add data visualization for exploratory analysis  
* \[ \] Include SHAP values for explainability  
* \[ \] Create batch prediction functionality  
* \[ \] Add model retraining capability through UI

## **🤝 Contributing**

Contributions are welcome\! Please feel free to submit a Pull Request.

1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request

## **📝 License**

This project is licensed under the MIT License \- see the LICENSE file for details.

## **🙏 Acknowledgments**

* Kaggle for providing the Titanic dataset  
* Streamlit for the amazing web framework  
* scikit-learn for machine learning tools

## **📧 Contact**

Your Name \- your.email@example.com

Project Link: [https://github.com/yourusername/titanic-decision-tree](https://github.com/yourusername/titanic-decision-tree)

---

Made with ❤️ and Python

