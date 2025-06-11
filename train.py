import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib 
import os

# ==== Load Dataset ====
df = pd.read_csv("D:\SLEEP_DISORDER\Data\Sleep_health_and_lifestyle_dataset.csv")

# ==== Preprocessing ====
df['Sleep Disorder'].fillna('None', inplace=True)
df.drop('Person ID', axis=1, inplace=True)
df['systolic_bp'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
df['diastolic_bp'] = df['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))
df.drop('Blood Pressure', axis=1, inplace=True)
df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')

# ==== Encode Target ====
target = 'Sleep Disorder'
X = df.drop(target, axis=1)
y = df[target]

# ==== Identify Column Types ====
cat_col = ['Gender', 'Occupation', 'BMI Category']
num_col = [col for col in X.columns if col not in cat_col]

# ==== Train-Test Split ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==== Column Transformer Pipeline ====
transform = ColumnTransformer(
    transformers=[
        ("cat_encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("scaler", StandardScaler(), num_col),
    ]
)

# ==== Full Pipeline ====
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ]
)

# ==== Train Model ====
pipe.fit(X_train, y_train)

# ==== Predict & Evaluate ====
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(f"✅ Accuracy: {round(accuracy, 4)}")
print(f"✅ F1 Score: {round(f1, 4)}")

# ==== Create Results Folder ====
os.makedirs("Results", exist_ok=True)
os.makedirs("Model", exist_ok=True)

# ==== Save Confusion Matrix ====
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Sleep Disorder Prediction")
plt.savefig("Results/sleep_model_cm.png", dpi=120)

# ==== Write Metrics to File ====
with open("Results/sleep_metrics.txt", "w") as f:
    f.write(f"Accuracy = {round(accuracy, 4)}\n")
    f.write(f"F1 Score = {round(f1, 4)}\n")

# ==== Save the Model ====
joblib.dump(pipe, "Model/sleep_pipeline.joblib")

print("🎉 Model and evaluation artifacts saved.")
