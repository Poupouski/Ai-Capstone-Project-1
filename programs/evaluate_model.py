import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from load_data import load_dataset

# Model loading
model = tf.keras.models.load_model("cnn_image_classifier.h5")

# Dataset loading
_, X_test, _, y_test = load_dataset()

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Precision on the test set: {test_acc:.4f}")



# Predicting test images class
y_pred_prob = model.predict(X_test)  # out Probability
y_pred = (y_pred_prob > 0.5).astype("int32")  # Treshold at 0.5 to convert to 0 or 1

# Generate and print classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Printing confusion matrix 
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "AI-Generated"], yticklabels=["Real", "AI-Generated"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()