# train.py

import dvc.api
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.decomposition import PCA
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from tqdm import tqdm


def get_model(model_name: str) -> (object, dict):
    """
    Get model object
    :param model_name: name of the model
    :return: model object and hyperparameter search space
    """

    if model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression

        # define search space for Logistic Regression
        search_space = {"clf__C": Real(1e-3, 1e3, prior="log-uniform")}
        return LogisticRegression(), search_space
    elif model_name == "SVC":
        from sklearn.svm import SVC

        # define search space for SVM classifier
        search_space = {
            "clf__C": Real(1e-3, 1e3, prior="log-uniform"),
            "clf__gamma": Real(1e-3, 10.0, prior="log-uniform"),
            "clf__degree": Integer(1, 3),
            "clf__kernel": Categorical(["linear", "poly", "rbf"]),
        }

        return SVC(), search_space
    else:
        raise ValueError(f"Model {model_name} not found")


def tune_model_pipeline(
    params: dict, model_pipeline: Pipeline, search_space: dict, train: pd.DataFrame
) -> BayesSearchCV:
    """
    Tune the model
    :param params: params for model tuning
    :param model_pipeline: ML model pipeline of interest
    :param search_space: hyperparameter search space
    :param train: train data
    :return: search algorithm with optimal score and best params
    """

    X_train, y_train = (
        np.array([vec for vec in train.sentence_embeddings.values]),
        train[params["target_name"]],
    )

    # define search algorithm
    cv = RepeatedStratifiedKFold(
        n_splits=params["tune"]["n_splits"], n_repeats=params["tune"]["n_repeats"]
    )
    search_algorithm = BayesSearchCV(
        model_pipeline,
        search_space,
        n_iter=params["tune"]["n_iter"],
        cv=cv,
        optimizer_kwargs={"base_estimator": "GP"},
    )

    overdone_control = DeltaYStopper(delta=params["tune"]["overdone_delta"])
    time_limit_control = DeadlineStopper(total_time=params["tune"]["time_limit"])

    search_algorithm.fit(
        X_train, y_train, callback=[overdone_control, time_limit_control]
    )

    return search_algorithm


def train_model_pipeline(
    params: dict, model_pipeline: Pipeline, train: pd.DataFrame
) -> Pipeline:
    """
    Train the model
    :param params: params for model training
    :param model_pipeline: ML model of interest
    :param train: train data
    :return: model pipeline
    """

    X_train, y_train = (
        np.array([vec for vec in train.sentence_embeddings.values]),
        train[params["target_name"]],
    )
    model_pipeline.fit(X_train, y_train)

    return model_pipeline


def save_model(repo_path: Path, params: dict, model_pipelines: dict) -> None:
    """
    Save the model pipeline
    :param repo_path: path to repo
    :param params: params for model training
    :param model_pipeline: model pipeline
    :return: None
    """

    pickle.dump(model_pipelines, open(repo_path / params["model_data_path"], "wb"))
    pickle.dump(model_pipelines, open(repo_path / params["model_result_path"], "wb"))


def main(repo_path: Path) -> None:
    """
    Main function
    :param repo_path: path to repo
    :return: None
    """

    # load params from DVC specific to train and evaluation
    params = dvc.api.params_show(stages=["preprocess", "train"])

    # set random seed
    np.random.seed(params["seed"])

    # load train data for each merchant
    train_data = {}
    for account_id in tqdm(params["account_ids"]):
        train_data[account_id] = pd.read_parquet(
            repo_path / f"{params['train_path']}_{account_id}"
        )

    # get or set experiment if not exists
    experiment = mlflow.set_experiment(params["experiment_name"])

    # Train model pipelines on the test set for each merchant
    # Track training specific params for the experiment
    model_pipelines = {}

    for account_id, train in tqdm(train_data.items()):
        with mlflow.start_run(
            experiment_id=experiment.experiment_id, run_name=f"training_{account_id}"
        ):
            mlflow.log_param("seed", params["seed"])
            mlflow.log_param("data_version", params["data_version"])
            mlflow.log_param("min_tag_cnt", params["min_tag_cnt"])
            mlflow.log_param("embedding_pooling_type", params["embedding_pooling_type"])
            mlflow.log_param("pca_n_components", params["pca_n_components"])

            model_pipelines[account_id] = {}

            for model_name in params["models"]:
                with mlflow.start_run(
                    experiment_id=experiment.experiment_id,
                    run_name=model_name,
                    nested=True,
                ):
                    model, search_space = get_model(model_name)
                    # setup a model pipeline
                    # PCA is used to reduce and alleviate the curse of dimensionality for linear classifiers
                    model_pipeline = Pipeline(
                        [
                            ("pca", PCA(n_components=params["pca_n_components"])),
                            ("clf", model),
                        ]
                    )
                    # tune model with hyperparameter search space

                    search_algorithm = tune_model_pipeline(
                        params, model_pipeline, search_space, train
                    )
                    mlflow.log_params(search_algorithm.best_params_)

                    # train model with optimal hyperparameters
                    model_pipelines[account_id][model_name] = train_model_pipeline(
                        params, search_algorithm.best_estimator_, train
                    )

    # save model pipeline for each merchant
    save_model(repo_path, params, model_pipelines)


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
