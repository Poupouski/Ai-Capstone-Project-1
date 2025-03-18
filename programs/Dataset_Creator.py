import pandas as pd
import os

# Define the working directory
current_dir = os.getcwd()  # Get the current directory path
csv_path = os.path.join(current_dir, "dataset_metadata.csv")  # Create absolute path

# Get file names and assing label
real_images = os.listdir("dataset/real_images")
ai_images = os.listdir("dataset/ai_generated")

data = []

# Add real pictures 
for img in real_images:
    data.append(["real_images/" + img, 0])  # 0 = Real Image

# Add AI pictures
for img in ai_images:
    data.append(["ai_generated/" + img, 1])  # 1 = AI image

# Save dataset
df = pd.DataFrame(data, columns=["Image_Path", "Label"])
df.to_csv(csv_path, index=False)  

print(f"Dataset structuré et prêt pour l'entraînement !\nFichier créé : {csv_path}")
