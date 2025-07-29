import os
import glob
import json
import pandas as pd
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
    if dataset_name == 'weed':
        # zip_file = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/weed_object_detection.zip"
        # csv_path = kagglehub.dataset_download("jaidalmotra/weed-detection")
        # download_dir = "./weed_object_detection"
        csv_path = kagglehub.dataset_download("jaidalmotra/weed-detection")
        train = csv_path + '/train'
        test = csv_path + '/test'
        csv_path = [train, test] 

    elif dataset_name == 'SkyFusion':
        path = kagglehub.dataset_download("kailaspsudheer/tiny-object-detection")
        path = path + '/SkyFusion'
        train = path + '/train'
        valid = path + '/valid'
        test = path + '/test'
        csv_path = [train, valid, test]
    return csv_path

def coco_to_dataframe(json_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    
    images_df = pd.DataFrame(images)
    annotations_df = pd.DataFrame(annotations)
    categories_df = pd.DataFrame(categories)
    
    categories_map = {row['id']: row['name'] for _, row in categories_df.iterrows()}
    annotations_df['category_name'] = annotations_df['category_id'].map(categories_map)
    
    merged_df = pd.merge(annotations_df, images_df, left_on='image_id', right_on='id', suffixes=('_annotation', '_image'))
    merged_df = merged_df.drop(columns=['id_annotation', 'id_image', 'image_id', 'category_id'], errors='ignore')
    # 컬럼 이름 변경
    column_mapping = {
        'file_name': 'image',
        'bbox': 'label'
    }
    merged_df = merged_df.rename(columns=column_mapping)
    return merged_df