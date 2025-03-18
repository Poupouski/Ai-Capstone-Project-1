import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from load_data import load_dataset

# Load dataset
X, X_test, y, y_test = load_dataset()
X = np.array(X)  # Ensure X is a NumPy array
y = np.array(y)  # Ensure y is a NumPy array

# Define K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
acc_per_fold = []

for train_idx, val_idx in kfold.split(X, y):
    print(f"\nTraining Fold {fold_no}...")

    # Split data into train and validation sets
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Define CNN Model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # Evaluate the model
    scores = model.evaluate(X_val, y_val, verbose=0)
    acc_per_fold.append(scores[1] * 100)

    print(f"Fold {fold_no} - Accuracy: {scores[1] * 100:.2f}%")
    fold_no += 1

# Print final results
print("\nCross-validation results:")
print(f"Mean Accuracy: {np.mean(acc_per_fold):.2f}%")
print(f"Standard Deviation: {np.std(acc_per_fold):.2f}%")

print(f'ADD: Average CNN Accuracy: {np.mean(acc_per_fold):.2f}% ± {np.std(acc_per_fold):.2f}%')