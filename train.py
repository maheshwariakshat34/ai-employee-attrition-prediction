import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print("\nTraining models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained successfully")


results = {}

print("\nModel Evaluation\n")

for name, model in models.items():
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("=" * 50)
    print(name)
    print("=" * 50)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print("Accuracy:", results[best_model_name])

os.makedirs("models", exist_ok=True)

model_path = "models/employee_attrition_model.pkl"

joblib.dump(best_model, model_path)

print("\nModel saved to:", model_path)