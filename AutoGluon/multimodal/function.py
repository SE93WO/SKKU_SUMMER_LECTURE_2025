import os
import glob
import kagglehub
from tqdm import tqdm
import pandas as pd

def generate_path(base_path, model_name):
    base_model_path = os.path.join(base_path, model_name)
    unique_model_path = base_model_path
    counter = 1

    while os.path.exists(unique_model_path):
        unique_model_path = f"{base_model_path}_{counter}"
        counter += 1

    return unique_model_path

def Get_Images_columns(data_type):
    base_path = './pet_finder/'
    if data_type == 'train':
        image_paths = os.listdir(base_path + 'train_images')
        
        data = pd.read_csv(f'{base_path}train/train.csv', index_col=0)
    elif data_type == 'test':
        image_paths = os.listdir(base_path + 'test_images')
        data = pd.read_csv(f'{base_path}test/test.csv', index_col=0)
        
    columns = list(data.columns)
    columns.append('Images')
    df = pd.DataFrame(columns=columns)
    for path in tqdm(image_paths):
        petId = path.split('-')[0]
        filtered_row = list(data[data['PetID'] == petId].values[0])
        filtered_row.append(f'{base_path}{data_type}_images/{path}')
        df.loc[len(df)] = filtered_row
    return df