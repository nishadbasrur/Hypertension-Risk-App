import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("framingham.csv")

# Feature engineering
df['pulsePressure'] = df['sysBP'] - df['diaBP']
df['smokeIntensity'] = df['currentSmoker'] * df['cigsPerDay']

# Define features and label
features = [
    'male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
    'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate',
    'pulsePressure', 'smokeIntensity'
]
X = df[features]
y = df['prevalentHyp']

# Handle missing values if any (optional safety step)
X = X.fillna(X.mean())

# Scale inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully.")
