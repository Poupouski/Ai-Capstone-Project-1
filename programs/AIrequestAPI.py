import requests
import os
import time

# API Stable Diffusion (Hugging Face)
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
HEADERS = {"Authorization": " ... "}

#  Generate pictures
def generate_stable_diffusion_images(prompt, count=50):
    
    folder_path = f"dataset/ai_generated/{prompt.replace(' ', '_')}"
    os.makedirs(folder_path, exist_ok=True)

    for i in range(count):
        print(f"Génération {i+1}/{count} : {prompt}")

        response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})

        if response.status_code != 200:
            print(f"Erreur {response.status_code} : {response.text}")
            break

        img_data = response.content
        file_path = f"{folder_path}/{prompt.replace(' ', '_')}_{i}.jpg"

        with open(file_path, "wb") as f:
            f.write(img_data)

        print(f"Image enregistrée : {file_path}")

        time.sleep(2) 

    print(f"Génération terminée pour {prompt} ({count} images)")

# Picture categories
categories_ai = [
    "futuristic city skyline at night",
    "hyper-realistic portrait of a person",
    "cyberpunk street view",
    "fantasy landscape with dragons",
    "dreamlike scenery with floating islands"
]

# Start to generate
for category in categories_ai:
    generate_stable_diffusion_images(category, count=125)
