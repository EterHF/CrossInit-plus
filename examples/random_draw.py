import pandas as pd
import random
import os
import shutil


def select_unique_breeds(data):
    unique_breeds = data['breed'].unique()
    selected_pairs = []
    
    for breed in unique_breeds:
        breed_ids = data[data['breed'] == breed]['id'].tolist()
        selected_id = random.choice(breed_ids)
        selected_pairs.append((selected_id, breed))
    
    return selected_pairs

file_path = './labels.csv'
try:
    data = pd.read_csv(file_path)
    preview = data.head()
except pd.errors.ParserError:
    preview = "The file is not in a proper CSV format."
except Exception as e:
    preview = str(e)

selected_id_breed_pairs = select_unique_breeds(data)


folder_a_path = './dog_dataset'
folder_b_path = './selected_photos'

if not os.path.exists(folder_b_path):
    os.makedirs(folder_b_path)

for idx, breed in selected_id_breed_pairs:
    photo_name_0 = idx + '.jpg'
    photo_name_1 = breed + '.jpg'

    photo_path = os.path.join(folder_a_path, photo_name_0)
    
    if os.path.exists(photo_path):
        destination_path = os.path.join(folder_b_path, photo_name_1)
        shutil.copy2(photo_path, destination_path)
    else:
        print(f'Photo {photo_name} not found in {folder_a_path}')

