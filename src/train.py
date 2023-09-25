# train.py

import pandas as pd
import numpy as np
import pickle

import dvc.api
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
    elif model_name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()
    else:
        raise ValueError(f"Model {model_name} not found")


def train_model_pipeline(params, model, train_data, tags):
    """
    Train the model
    :param params: params for model training
    :param model: ML model of interest
    :param train: train data
    :param tags: tags of interest for model training
    :return: model pipeline
    """

    model_pipeline = Pipeline([
        ('pca', PCA(n_components=params["pca_n_components"])),
        ('clf', model)
    ])
    
    for tag in tqdm(tags[:5]):
        train = train_data[tag]
        X_train, y_train = np.array([vec for vec in train.sentence_embeddings.values]), train[tag]  
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


def main(repo_path):
    """
    Main function
    :param repo_path: path to repo
    :return: None
    """

    print("repo_path:", repo_path)
    params = dvc.api.params_show(stages="train")
    print("params:")
    print(params)

    print("Loading tags...")
    tags = np.load(repo_path / params["tags_path"])

    print("Loading train data...")
    train_data = {}
    for tag in tqdm(tags):
        train_data[tag] = pd.read_parquet(repo_path / params["train_path"] / f"{tag.replace('/', '_')}")

    print("Training model pipelines...")
    model_pipelines = {}
    for model_name in params["models"]:
        model = get_model(model_name)
        model_pipelines[model_name] = train_model_pipeline(params, model, train_data, tags)

    print("Saving model pipelines...")
    save_model(repo_path, params, model_pipelines)


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
