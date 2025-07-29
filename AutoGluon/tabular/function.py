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
    if dataset_name == 'titanic':
        path = kagglehub.dataset_download("danielwe14/titanic-dataset-with-solution")
        csv_path = glob.glob(f"{path}/**/*.csv")
    elif dataset_name == 'kepler':
        path = kagglehub.dataset_download("nasa/kepler-exoplanet-search-results")
        csv_path = glob.glob(f"{path}/*.csv")
    elif dataset_name == 'news_feed':
        path = current_path + '/Ranking-social-media-news-feed/'
        #os.systen('source .')
        os.system(f'git clone https://github.com/SamBelkacem/Ranking-social-media-news-feed.git')
        csv_path = glob.glob(f"{path}/*.csv")
    print(csv_path)
    return csv_path