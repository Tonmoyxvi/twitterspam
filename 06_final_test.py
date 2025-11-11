import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("imagens_step6", exist_ok=True)

X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = joblib.load("data/split_data.joblib")
rf_tuned = joblib.load("data/model_rf_tuned.joblib")

print("=== FINAL TEST RESULTS ===")
y_pred = rf_tuned.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Final Model - Test Confusion Matrix')
plt.savefig("imagens_step6/final_confusion_matrix.png")
plt.close()

y_proba = rf_tuned.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
plt.plot(fpr, tpr, label=f'Final RF (AUC={auc:.3f})')
plt.legend(); plt.title('Final ROC Curve')
plt.savefig("imagens_step6/final_roc.png")
plt.close()

print("✅ Final evaluation complete. Results saved to imagens_step6/")
