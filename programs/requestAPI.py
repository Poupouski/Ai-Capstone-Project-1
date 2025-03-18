import requests
import os
import time

ACCESS_KEY = " "

def download_images(query, total_images=50, per_page=10):
    
    save_path = "dataset/real_images"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    downloaded = 0
    page = 1

    while downloaded < total_images:
        url = f"https://api.unsplash.com/search/photos?query={query}&page={page}&per_page={per_page}&client_id={ACCESS_KEY}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Erreur API Unsplash ({response.status_code}) : {response.json()}")
            break

        data = response.json()
        if "results" not in data:
            print(f" Aucune image trouvée pour {query}.")
            break

        for img in data["results"]:
            if downloaded >= total_images:
                break

            img_url = img["urls"]["regular"]
            img_data = requests.get(img_url)

            if img_data.status_code == 200:
                file_path = os.path.join(save_path, f"{query}_{downloaded}.jpg")
                with open(file_path, "wb") as f:
                    f.write(img_data.content)
                downloaded += 1
                print(f"{downloaded}/{total_images} images téléchargées pour {query}")
                print(f"Images enregistrées dans : {os.path.abspath(save_path)}")

            else:
                print(f"Impossible de télécharger l'image : {img_url}")

        page += 1
        time.sleep(3)  

    print(f"Téléchargement terminé pour {query} ({downloaded} images)")

categories = ["countryside"]
for category in categories:
    download_images(category, total_images=70, per_page=10)
