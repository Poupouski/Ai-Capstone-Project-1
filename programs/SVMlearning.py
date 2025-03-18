import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from load_data import load_dataset


# Loading data
X_train, X_test, y_train, y_test = load_dataset()

# Data shaping 
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Data normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definition and training of SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Classification report
class_report = classification_report(y_test, y_pred, output_dict=True)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "AI-Generated"],
            yticklabels=["Real", "AI-Generated"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# --- Visualization of F1 scores, precision, recall ---
metrics = ["precision", "recall", "f1-score"]
labels = list(class_report.keys())[:-3]  

for metric in metrics:
    values = [class_report[label][metric] for label in labels]
    plt.bar(labels, values, label=metric)

plt.ylim(0, 1)
plt.xlabel("Classes")
plt.ylabel("Score")
plt.title("Precision, Recall & F1-Score per Class")
plt.legend()
plt.show()
