# evaluate.py

import pandas as pd
import numpy as np
import pickle

import dvc.api
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")



def inference_and_evaluate(experiment, params, model_pipeline, test_data, tags, model_name):
    """
    Evaluate the model pipeline on the test set
    :param experiment: evalution experiment
    :param params: params for model training
    :param model_pipeline: the model pipeline to evaluate
    :param test_data: the test data
    :param tags: the tags of interest
    :param model_name: the name of the model
    :return: None
    """
    
    result_list = []
    
    for tag in tqdm(tags):
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"tag: {str(tag)}", nested=True):

            test = test_data[tag]
            X_test, y_test = np.array([vec for vec in test.sentence_embeddings.values]), test[tag]
            y_pred = model_pipeline.predict(X_test)

            f1__score = f1_score(y_test, y_pred)
            pr_score = precision_score(y_test, y_pred)
            rec_score = recall_score(y_test, y_pred)

            mlflow.log_metric("f1_score", f1__score)
            mlflow.log_metric("precision_score", pr_score)
            mlflow.log_metric("recall_score", rec_score)
            
            result_list.append({
                "account_ids": ','.join(params["account_ids"]),
                "model_name": model_name,
                "tag": tag,
                "f1_score": f1__score,
                "precision": pr_score,
                "recall": rec_score
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
    results.to_csv(repo_path / params['result_path_track'], index=False)


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

    experiment = mlflow.set_experiment(params["experiment_name"])

    print("Inference model pipelines...")
    result_list = []

    for model_name in params["models"]:
        print("Inference model:", model_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=f"Evaluation of {model_name}"):

            mlflow.log_param("seed", params["seed"])
            mlflow.log_param("data_version", params["data_version"])
            mlflow.log_param("account_ids", params["account_ids"])
            mlflow.log_param("embedding_pooling_type", params["embedding_pooling_type"])
            mlflow.log_param("preprocess_type", params["preprocess_type"])
            mlflow.log_param("model", model_name)
            mlflow.log_param("pca_n_components", params["pca_n_components"])
            
            result_list.append(
                inference_and_evaluate(experiment, params, model_pipelines[model_name], test_data, tags, model_name)
            )

    print("Saving results...")
    save_results(repo_path, params, pd.concat(result_list))


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
