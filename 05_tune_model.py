import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = joblib.load("data/split_data.joblib")

param_grid = {
    'max_depth': [3, 5, 7, None],
    'n_estimators': [50, 100, 200],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced']
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

rf_best = grid.best_estimator_
val_f1 = f1_score(y_val, rf_best.predict(X_val))
print(f"✅ Best params: {grid.best_params_} | Validation F1: {val_f1:.4f}")

joblib.dump(rf_best, "data/model_rf_tuned.joblib")
print("✅ Tuned RandomForest model saved to data/model_rf_tuned.joblib")
