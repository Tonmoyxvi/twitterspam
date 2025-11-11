import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("imagens", exist_ok=True)

X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = joblib.load("data/split_data.joblib")
lr, rf, best_model, best_name = joblib.load("data/models_initial.joblib")

def evaluate_model(model, X, y, name, dataset="Validation"):
    print(f"\n{name} - {dataset} Report:\n", classification_report(y, model.predict(X)))
    cm = confusion_matrix(y, model.predict(X))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - {dataset} Confusion Matrix')
    plt.savefig(f"imagens/confusion_matrix_{name.lower()}_{dataset.lower()}.png")
    plt.close()

    y_proba = model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc = roc_auc_score(y, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    plt.legend(); plt.title(f'ROC - {dataset}'); plt.savefig(f"imagens/roc_{name.lower()}_{dataset.lower()}.png")
    plt.close()

evaluate_model(lr, X_val, y_val, "LogisticRegression")
evaluate_model(rf, X_val, y_val, "RandomForest")

print("✅ Evaluation complete. Confusion matrices and ROC curves saved in /imagens/")


def compare_confusion_matrices(models, X, y, names, dataset="Validation"):
    plt.figure(figsize=(12, 5))

    for i, (model, name) in enumerate(zip(models, names), 1):
        cm = confusion_matrix(y, model.predict(X))
        plt.subplot(1, len(models), i)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} - {dataset}')
        plt.xlabel('Predicted')
        plt.ylabel('True')

    plt.tight_layout()
    plt.savefig(f"imagens/confusion_matrix_comparison_{dataset.lower()}.png")
    plt.close()
    print(f"✅ Confusion Matrix comparison saved to imagens/confusion_matrix_comparison_{dataset.lower()}.png")


import numpy as np

feature_names = vectorizer.get_feature_names_out()
importances = rf.feature_importances_
top_idx = np.argsort(importances)[-20:][::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_idx)), importances[top_idx])
plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
plt.gca().invert_yaxis()
plt.title("Top 20 Important Words for Spam Detection")
plt.xlabel("Feature Importance")
plt.savefig("imagens/feature_importance.png")
plt.show()

coef = lr.coef_.ravel()
feature_names = np.array(vectorizer.get_feature_names_out())

# Get top positive (spam) and negative (ham) coefficients
top_positive_idx = np.argsort(coef)[-20:][::-1]
top_negative_idx = np.argsort(coef)[:20]

plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in top_positive_idx], coef[top_positive_idx], color='green')
plt.gca().invert_yaxis()
plt.title("Top 20 Words Indicating SPAM (Logistic Regression)")
plt.xlabel("Coefficient Weight")
plt.tight_layout()
plt.savefig("imagens/feature_importance_logreg_spam.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in top_negative_idx], coef[top_negative_idx], color='red')
plt.gca().invert_yaxis()
plt.title("Top 20 Words Indicating HAM (Logistic Regression)")
plt.xlabel("Coefficient Weight")
plt.tight_layout()
plt.savefig("imagens/feature_importance_logreg_ham.png")
plt.show()

