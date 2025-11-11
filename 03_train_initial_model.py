import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = joblib.load("data/split_data.joblib")

models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
}

best_model, best_name, best_f1 = None, "", 0

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds)
    acc = accuracy_score(y_val, preds)
    print(f"{name} → Accuracy: {acc:.4f}, F1: {f1:.4f}")

    if f1 > best_f1:
        best_model, best_name, best_f1 = model, name, f1

joblib.dump((models['LogisticRegression'], models['RandomForest'], best_model, best_name), "data/models_initial.joblib")
print(f"✅ Best model: {best_name} (F1 = {best_f1:.4f}) saved in data/models_initial.joblib")
