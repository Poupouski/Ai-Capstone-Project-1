from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# Predict on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class indices
y_true = np.argmax(y_test, axis=1)  # If one-hot encoded, convert to class indices

# Print classification report
print(classification_report(y_true, y_pred_classes))
