import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv("iris.csv")

# Features & target
X = df.drop("species", axis=1)
y = df["species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Decision Tree Model
# -----------------------------
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nDecision Tree Report:\n", classification_report(y_test, y_pred_dt))

# -----------------------------
# SVM Model
# -----------------------------
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Report:\n", classification_report(y_test, y_pred_svm))

# Save SVM Model
pickle.dump(svm_model, open("svm_model.pkl", "wb"))
