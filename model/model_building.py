import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score

# 1. Load the dataset
# wdbc.data does not have a header, so we define names manually
# 1. Load the dataset
column_names = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df = pd.read_csv('wdbc.data', names=column_names, header=None)

# 2. Feature Selection (Using the specific names from your requirements)
# We map the generic feature names to the ones required
df = df.rename(columns={
    'feature_1': 'radius_mean',
    'feature_2': 'texture_mean',
    'feature_3': 'perimeter_mean',
    'feature_4': 'area_mean',
    'feature_5': 'smoothness_mean'
})

selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
X = df[selected_features]

# --- THE FIX FOR MULTICLASS ERROR ---
# Clean the diagnosis column: strip whitespace and ensure only M and B exist
y = df['diagnosis'].str.strip() 

# Force binary encoding: Malignant = 1, Benign = 0
y = y.map({'M': 1, 'B': 0})

# Check if any NaNs were created (happens if there's a typo in the data file)
if y.isnull().any():
    print("⚠️ Warning: Found unexpected values in diagnosis column. Dropping rows...")
    valid_indices = y.dropna().index
    X = X.loc[valid_indices]
    y = y.dropna()

# Now proceed to split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature Scaling (Mandatory for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Implement SVM Algorithm
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluation
y_pred = model.predict(X_test_scaled)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the trained model and scaler to disk
# Ensure the 'model' directory exists
if not os.path.exists('model'):
    os.makedirs('model')

model_data = {
    'classifier': model,
    'scaler': scaler,
    'features': selected_features
}

with open('model/breast_cancer_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n✅ Model and Scaler saved to 'model/breast_cancer_model.pkl'")

# 7. Demonstration of Reloading (Part A, Requirement 7)
with open('model/breast_cancer_model.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    reloaded_model = saved_data['classifier']
    
print("✅ Verification: Saved model reloaded successfully.")