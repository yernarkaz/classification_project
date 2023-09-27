# evaluate.py

import dvc.api
import mlflow
import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from tqdm import tqdm


def inference_and_evaluate(
    params: dict,
    account_id: str,
    model_pipeline: dict,
    test: pd.DataFrame,
    model_name: str,
) -> dict:
    """
    Inference and evaluate the model pipeline on the test set
    :param params: params for model training
    :param account_id: the account id of the merchant
    :param model_pipeline: the model pipeline to evaluate
    :param test: the test data
    :param model_name: the name of the model
    :return: dict of metrics
    """

    X_test, y_test = (
        np.array([vec for vec in test.sentence_embeddings.values]),
        test[params["target_name"]],
    )
    y_pred = model_pipeline.predict(X_test)

    f1_micro_score = f1_score(y_test, y_pred, average="micro")
    f1_macro_score = f1_score(y_test, y_pred, average="macro")
    f1_weighted_score = f1_score(y_test, y_pred, average="weighted")
    pr_score = precision_score(y_test, y_pred, average="weighted")
    rec_score = recall_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("f1_micro_score", f1_micro_score)
    mlflow.log_metric("f1_macro_score", f1_macro_score)
    mlflow.log_metric("f1_weighted_score", f1_weighted_score)
    mlflow.log_metric("precision_score", pr_score)
    mlflow.log_metric("recall_score", rec_score)

    return {
        "account_id": account_id,
        "model_name": model_name,
        "f1_micro_score": f1_micro_score,
        "f1_macro_score": f1_macro_score,
        "f1_weighted_score": f1_weighted_score,
        "precision": pr_score,
        "recall": rec_score,
        "classification_report": classification_report(y_test, y_pred),
    }


def load_model(repo_path: Path, params: dict) -> dict:
    """
    Load the model pipelines
    :param repo_path: path to repo
    :param params: params for model training
    :return: model_pipeline: dict of model pipelines
    """

    return pickle.load(open(repo_path / params["model_path"], "rb"))


def save_results(repo_path: Path, params: dict, results: pd.DataFrame) -> None:
    """
    Save performance results of model pipelines
    :param repo_path: path to repo
    :param params: params for model training
    :return: None
    """

    results.to_csv(repo_path / params["result_path"], index=False)
    results.to_csv(repo_path / params["result_path_track"], index=False)


def main(repo_path: Path) -> None:
    """
    Main function
    :param repo_path: path to repo
    :return: None
    """

    # load params from DVC specific to train and evaluation
    params = dvc.api.params_show(stages=["train", "evaluate"])

    # load test data for each merchant
    test_data = {}
    for account_id in tqdm(params["account_ids"]):
        test_data[account_id] = pd.read_parquet(
            repo_path / f"{params['test_path']}_{account_id}"
        )

    # load model pipelines
    model_pipelines = load_model(repo_path, params)

    # get or set experiment if not exists
    experiment = mlflow.set_experiment(params["experiment_name"])

    # Inference and evaluate model pipelines on the test set for each merchant
    # Track evaluation metrics for experiment
    result_list = []

    for account_id, test in tqdm(test_data.items()):
        with mlflow.start_run(
            experiment_id=experiment.experiment_id, run_name=f"evaluating_{account_id}"
        ):
            for model_name in params["models"]:
                with mlflow.start_run(
                    experiment_id=experiment.experiment_id,
                    run_name=model_name,
                    nested=True,
                ):
                    res = inference_and_evaluate(
                        params,
                        account_id,
                        model_pipelines[account_id][model_name],
                        test,
                        model_name,
                    )
                    result_list.append(res)

    print("Saving results...")
    save_results(repo_path, params, pd.DataFrame(result_list))


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
