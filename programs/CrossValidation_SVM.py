import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from load_data import load_dataset

# Load dataset
X, X_test, y, y_test = load_dataset()

# Flatten images
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# K-Fold Cross-Validation setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
accuracy_per_fold = []

for train_idx, val_idx in kfold.split(X, y):
    print(f"\nTraining Fold {fold_no}...")

    # Split data into train and validation sets
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Define and train SVM model
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_model.fit(X_train, y_train)

    # Predictions
    y_pred = svm_model.predict(X_val)

    # Evaluate the model
    accuracy = accuracy_score(y_val, y_pred)
    accuracy_per_fold.append(accuracy)

    print(f"Fold {fold_no} - Accuracy: {accuracy:.4f}")
    fold_no += 1

# Print final results
print("\nCross-validation results:")
print(f"Mean Accuracy: {np.mean(accuracy_per_fold):.4f}")
print(f"Standard Deviation: {np.std(accuracy_per_fold):.4f}")

# Final model training on full dataset
final_svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
final_svm_model.fit(X, y)

# Final evaluation on test set
y_test_pred = final_svm_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")

# Classification Report
print("Final Classification Report:\n", classification_report(y_test, y_test_pred))
