# train.py

import pandas as pd
import numpy as np
import pickle

import dvc.api
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from tqdm import tqdm


def get_model(model_name):
    """
        Get model object
        :param model_name: name of the model
        :return: model object
        """

    if model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression()
    elif model_name == "SVC":
        from sklearn.svm import SVC
        return SVC()
    else:
        raise ValueError(f"Model {model_name} not found")


def train_model_pipeline(params, model, train):
    """
    Train the model
    :param params: params for model training
    :param model: ML model of interest
    :param train: train data
    :return: model pipeline
    """

    model_pipeline = Pipeline([
        ('pca', PCA(n_components=params["pca_n_components"])),
        ('clf', model)
    ])
    
    X_train, y_train = np.array([vec for vec in train.sentence_embeddings.values]), train[params['target_name']]  
    model_pipeline.fit(X_train, y_train)

    return model_pipeline


def save_model(repo_path, params, model_pipelines):
    """
    Save the model pipeline
    :param repo_path: path to repo
    :param params: params for model training
    :param model_pipeline: model pipeline
    :return: None
    """

    pickle.dump(model_pipelines, open(repo_path / f"data/model_pipelines.pkl", "wb"))
    pickle.dump(model_pipelines, open(repo_path / f"results/models/model_pipelines.pkl", "wb"))


def main(repo_path):
    """
    Main function
    :param repo_path: path to repo
    :return: None
    """

    params = dvc.api.params_show(stages=["preprocess", "train"])

    print("Seeding random...")
    np.random.seed(params["seed"])

    print("Loading tags...")
    selected_tags = pickle.load(open(repo_path / params["tags_path"], 'rb'))

    print("Loading train data...")
    train_data = {}
    for account_id in tqdm(params['account_ids']):
        train_data[account_id] = pd.read_parquet(repo_path / f"{params['train_path']}_{account_id}")
    
    experiment = mlflow.set_experiment(params["experiment_name"])

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="training"):
        mlflow.log_param("seed", params["seed"])
        mlflow.log_param("data_version", params["data_version"])
        mlflow.log_param("min_tag_cnt", params["min_tag_cnt"])
        mlflow.log_param("embedding_pooling_type", params["embedding_pooling_type"])
        
        model_pipelines = {}

        for account_id, train in tqdm(train_data.items()):

            print(f"Training model pipelines for merchant: {account_id}...")
            model_pipelines[account_id] = {}

            for model_name in params["models"]:

                print("Training model:", model_name)

                with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"{model_name}_{account_id}", nested=True):
                    mlflow.log_param("pca_n_components", params["pca_n_components"])

                    model = get_model(model_name)
                    model_pipelines[account_id][model_name] = train_model_pipeline(params, model, train)

    print("Saving model pipelines...")
    save_model(repo_path, params, model_pipelines)


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
