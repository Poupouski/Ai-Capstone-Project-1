import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2

def load_dataset(csv_path="dataset_metadata.csv", img_size=(128, 128)):
    data_dir = "dataset/"
    df = pd.read_csv(csv_path)
    images, labels = [], []

    for index, row in df.iterrows():
        img_path = os.path.join(data_dir, row["Image_Path"].strip())
        if not os.path.exists(img_path):
            print(f"Image non trouvée : {img_path}")
            continue
        try:
            image = load_img(img_path, target_size=img_size)
            image = img_to_array(image) / 255.0
        except:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Impossible de lire l’image : {img_path}")
                continue
            image = cv2.resize(image, img_size)
            image = image.astype(np.float32) / 255.0
        
        images.append(image)
        labels.append(row["Label"])

    X = np.array(images)
    y = np.array(labels)

    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"Dataset chargé : {X_train.shape[0]} images pour l'entraînement, {X_test.shape[0]} pour le test.")
