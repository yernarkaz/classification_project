# evaluate.py

import pandas as pd
import numpy as np
import pickle

import dvc.api
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")



def inference_and_evaluate(params, model_pipeline, test_data, tags, model_name):
    """
    Evaluate the model pipeline on the test set
    :param params: params for model training
    :param model_pipeline: the model pipeline to evaluate
    :param test_data: the test data
    :param tags: the tags of interest
    :param model_name: the name of the model
    :return: None
    """
    
    result_list = []
    for tag in tqdm(tags):
        
        test = test_data[tag]
        X_test, y_test = np.array([vec for vec in test.sentence_embeddings.values]), test[tag]
        y_pred = model_pipeline.predict(X_test)
        
        result_list.append({
            "model_name": model_name,
            "tag": tag,
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        })

    return pd.DataFrame(result_list)


def load_model(repo_path, params):
    """
    Load the model pipelines
    :param repo_path: path to repo
    :param params: params for model training
    :return: model_pipeline: model pipeline
    """

    return pickle.load(open(repo_path / params['model_path'], "rb"))


def save_results(repo_path, params, results):
    """
    Save performance results of model pipelines
    :param repo_path: path to repo
    :param params: params for model training
    :return: None
    """

    results.to_csv(repo_path / params['result_path'], index=False)


def main(repo_path):
    """
    Main function
    :param repo_path: path to repo
    :return: None
    """

    print("repo_path:", repo_path)
    params = dvc.api.params_show(stages=["train", "evaluate"])
    print("params:")
    print(params)

    print("Loading tags...")
    tags = np.load(repo_path / params["tags_path"])

    print("Loading test data...")
    test_data = {}
    for tag in tqdm(tags):
        test_data[tag] = pd.read_parquet(repo_path / params["test_path"] / f"{tag.replace('/', '_')}")

    print("Loading model pipelines...")
    model_pipelines = load_model(repo_path, params)

    print("Inference model pipelines...")
    result_list = []
    for model_name in params["models"]:
        result_list.append(
            inference_and_evaluate(params, model_pipelines[model_name], test_data, tags, model_name)
        )

    print("Saving results...")
    save_results(repo_path, params, pd.concat(result_list))


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
