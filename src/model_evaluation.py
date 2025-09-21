# src/model_evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve


def evaluate_model(model, X_test, y_test):
 y_pred = model.predict(X_test)
 y_proba = model.predict_proba(X_test)[:,1]


 # Confusion Matrix
 cm = confusion_matrix(y_test, y_pred)
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
 plt.title('Confusion Matrix')
 plt.xlabel('Predicted')
 plt.ylabel('Actual')
 plt.show()


 # Classification Report
 print(classification_report(y_test, y_pred))


 # ROC Curve
 auc = roc_auc_score(y_test, y_proba)
 fpr, tpr, _ = roc_curve(y_test, y_proba)
 plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
 plt.plot([0,1],[0,1],'k--')
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 plt.title('ROC Curve')
 plt.legend()
 plt.show()


 print(f"ROC AUC Score: {auc:.4f}")
