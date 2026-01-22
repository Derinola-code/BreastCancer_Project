import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_breast_cancer

# 1. Load Data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target # 0: Malignant, 1: Benign

# 2. Feature Selection (Selecting 5 as per instructions)
features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
X = df[features]
y = df['diagnosis']

# 3. Preprocessing & Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Save Model and Scaler
joblib.dump(model, 'model/breast_cancer_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl') # Crucial for scaling new user input