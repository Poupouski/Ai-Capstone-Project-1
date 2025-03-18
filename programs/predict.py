import tensorflow as tf
import numpy as np
import cv2
import sys

# Model loading
model = tf.keras.models.load_model("cnn_image_classifier.h5")

def predict_image(img_path):
    img_size = (128, 128)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Erreur : Impossible de lire l’image {img_path}")
        return
    image = cv2.resize(image, img_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    # Prediction
    prediction = model.predict(image)[0][0]
    label = "AI generated image" if prediction > 0.5 else "Real image"
    print(f"Prediction : {label} (Score: {prediction:.4f})")

# Test with the picture argument
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python predict.py <chemin_de_l_image>")
    else:
        predict_image(sys.argv[1])
