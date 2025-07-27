import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from xgboost import XGBClassifier
import joblib

# Step 1: Load the dataset
data = pd.read_csv('spambase.data', header=None)
columns = [f'feature_{i}' for i in range(57)] + ['label']
data.columns = columns

# Step 2: Split features and labels
X = data.iloc[:, :-1]
y = data['label']

# Step 3: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# Step 5: Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the model
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# (Optional) Step 8: Predict a new custom input (simulate email)
sample_email = X_test[0].reshape(1, -1)
print("Predicted (0 = Not Spam, 1 = Spam):", model.predict(sample_email)[0])
