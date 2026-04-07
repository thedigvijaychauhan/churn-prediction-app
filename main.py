import pandas as pd
import matplotlib.pyplot as plt

# LOAD DATA
df = pd.read_csv("data/churn.csv")

# FIX COLUMN NAMES
df.columns = df.columns.str.strip()

print("Data Loaded Successfully")
print(df.head())

# DATA CLEANING
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(0, inplace=True)

# TARGET FIX (IMPORTANT)
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# ENCODING (exclude Churn)
df = pd.get_dummies(df, drop_first=True)

# SPLIT DATA
from sklearn.model_selection import train_test_split

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# MODEL TRAINING
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# PREDICTION
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("Accuracy:", accuracy_score(y_test, y_pred))

# GRAPH
y.value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.show()
import pickle

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved ✅")