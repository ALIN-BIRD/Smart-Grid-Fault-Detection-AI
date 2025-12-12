import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # For saving the trained model
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- CONFIGURATION ---
input_csv = 'final_model_training_data.csv'
model_save_path = 'grid_fault_ai_model.pkl'

print("--- STARTING AI TRAINING ---")

# 1. LOAD DATA
try:
    df = pd.read_csv(input_csv)
    print(f"Dataset Loaded. Total Samples: {len(df)}")
except FileNotFoundError:
    print(f"CRITICAL ERROR: '{input_csv}' not found.")
    sys.exit()

# 2. DATA VALIDATION (Robustness Check)
# Check for missing values (NaN) or Infinity
if df.isnull().values.any():
    print("Warning: Dataset contains missing values (NaN). Dropping them...")
    df = df.dropna()

# Check if we have enough data
if len(df) < 100:
    print("Error: Dataset is too small for meaningful training.")
    sys.exit()

# 3. PREPARE INPUTS (X) AND OUTPUTS (y)
# X = The 12 features (RMS and Peak values)
# y = The Fault Label (0, 1, 2...)
feature_cols = [col for col in df.columns if col not in ['Run_ID', 'Fault_Type', 'Label']]
X = df[feature_cols]
y = df['Label']

print(f"Training with {len(feature_cols)} features: {feature_cols}")

# 4. SPLIT DATA (Train vs Test)
# 80% used for training, 20% hidden for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training Set: {len(X_train)} samples")
print(f"Testing Set:  {len(X_test)} samples")

# 5. TRAIN THE MODEL
print("\nTraining Random Forest Model...")
# n_estimators=100 means we use 100 decision trees voting together
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# Add random noise to simulate sensor error (+/- 5%)
noise = np.random.normal(0, 0.05, X_train.shape) 
X_train_noisy = X_train * (1 + noise)

# Train on noisy data
clf.fit(X_train_noisy, y_train)


clf.fit(X_train, y_train)
print("Training Complete!")

# 6. EVALUATE PERFORMANCE
print("\n--- TEST RESULTS ---")
y_pred = clf.predict(X_test)

# Calculate Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")
print("(This means the AI correctly identified the fault type X% of the time)")

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. SAVE THE TRAINED MODEL
print(f"Saving trained model to '{model_save_path}'...")
joblib.dump(clf, model_save_path)
print("Model Saved Successfully.")

# 8. VISUALIZE RESULTS (Confusion Matrix)
# This shows exactly where the model got confused (if anywhere)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title(f'AI Fault Detection Results (Accuracy: {acc*100:.1f}%)')
plt.xlabel('Predicted Fault Type (0-11)')
plt.ylabel('Actual Fault Type (0-11)')
plt.show()