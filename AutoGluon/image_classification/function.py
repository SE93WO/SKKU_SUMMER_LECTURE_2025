import os
import glob
import kagglehub

def generate_path(base_path, model_name):
    base_model_path = os.path.join(base_path, model_name)
    unique_model_path = base_model_path
    counter = 1

    while os.path.exists(unique_model_path):
        unique_model_path = f"{base_model_path}_{counter}"
        counter += 1

    return unique_model_path


def download_dataset(dataset_name):
    current_path = os.getcwd()
    os.makedirs(current_path, exist_ok=True)
    if dataset_name == 'birds':
        path = kagglehub.dataset_download("percywei/cs56000-birds-data-v2")
        csv_path = glob.glob(f"{path}/*.csv")
    return csv_path 