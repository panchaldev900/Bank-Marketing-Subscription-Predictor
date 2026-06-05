import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load dataset

dataset_paths = ["bank.csv", "bank-full.csv", "bank_cleaned.csv"]
dataset_path = next((path for path in dataset_paths if os.path.exists(path)), None)
if dataset_path is None:
    raise FileNotFoundError(
        f"Dataset not found. Expected one of: {', '.join(dataset_paths)}"
    )

csv_kwargs = {"sep": ";"} if dataset_path.endswith("bank-full.csv") else {}
df = pd.read_csv(dataset_path, **csv_kwargs)

# Rename target

df.rename(columns={"y": "response"}, inplace=True)

# Target encoding

df["response_binary"] = df["response"].map({
"yes": 1,
"no": 0
})

# Remove leakage column

if "duration" in df.columns:
    df.drop("duration", axis=1, inplace=True)

# One-hot encoding

df = pd.get_dummies(df, drop_first=True)

# Remove response_yes if created

if "response_yes" in df.columns:
    df.drop("response_yes", axis=1, inplace=True)

X = df.drop("response_binary", axis=1)
y = df["response_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)
rf_score = rf.score(X_test_scaled, y_test)

# Train Logistic Regression
lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train_scaled, y_train)
lr_score = lr.score(X_test_scaled, y_test)

# Save both models
models = {
    "random_forest": {"model": rf, "accuracy": rf_score},
    "logistic_regression": {"model": lr, "accuracy": lr_score}
}

with open("models.pkl", "wb") as models_file:
    pickle.dump(models, models_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

with open("features.pkl", "wb") as features_file:
    pickle.dump(list(X.columns), features_file)

print(f"Models saved successfully!")
print(f"Random Forest Test Accuracy: {rf_score:.4f}")
print(f"Logistic Regression Test Accuracy: {lr_score:.4f}")
