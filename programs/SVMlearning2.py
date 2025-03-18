import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from load_data import load_dataset
from skimage.transform import resize

# Data loading
X_train, X_test, y_train, y_test = load_dataset()

# Size reduction to limit features
X_train = np.array([resize(img, (64, 64)) for img in X_train])
X_test = np.array([resize(img, (64, 64)) for img in X_test])

# from 64x64x3 to 12288 features per image
X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)  # Conversion en float32
X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)  # Conversion en float32

# Selection of a data subset (30% of the dataset for training)
X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=0.3, random_state=42)

print(f"Nombre d'échantillons après sous-échantillonnage: {X_train_sub.shape[0]}")

# Standardisation
scaler = StandardScaler()
X_train_sub = scaler.fit_transform(X_train_sub)
X_test = scaler.transform(X_test)

# SVM Model definition  
param_grid = {
    'C': [0.1, 1, 10],  
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  
    'kernel': ['linear', 'rbf', 'poly']  
}

svm = RandomizedSearchCV(SVC(), param_grid, n_iter=5, cv=3, verbose=2, n_jobs=-1)
svm.fit(X_train_sub, y_train_sub)

# Best parameters found 
print(f"Best parameters: {svm.best_params_}")

# Prediction on the test set
y_pred = svm.best_estimator_.predict(X_test)

# classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
