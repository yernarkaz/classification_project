# preprocess.py
import dvc.api
import json
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm


def preprocess_data_for_merchant(
    data: pd.DataFrame, account_id: str, params: dict
) -> (pd.DataFrame, pd.DataFrame, list):
    """
    Preprocess input data for merchant
    :param data: input data
    :param account_id: merchant _id
    :param params: dict of params
    :return: train and test data, list of selected tags
    """

    # slice data specific to merchant
    merchant_data = (
        data[data.account_id == account_id]
        .dropna(subset="sentence_embeddings")
        .reset_index(drop=True)
    )

    # aggregate by email_sentence_hashed
    label_sentence_agg = (
        merchant_data.groupby("email_sentence_hashed")
        .contact_reason.apply(lambda x: set(x))
        .to_frame("tag_set")
        .reset_index()
    )
    label_sentence_agg["tag_len"] = label_sentence_agg.tag_set.apply(lambda x: len(x))

    # filter by email_sentence_hashed that has only one contact_reason
    mask = merchant_data.email_sentence_hashed.isin(
        label_sentence_agg.loc[label_sentence_agg.tag_len == 1, "email_sentence_hashed"]
    )
    merchant_data = merchant_data[mask]

    # keep only tags that appear at least min_tag_cnt times
    target_cnt = merchant_data[params["target_name"]].value_counts()
    target_cnt = target_cnt[target_cnt >= params["min_tag_cnt"]]
    selected_tags = target_cnt.index.tolist()

    merchant_data = merchant_data[
        merchant_data[params["target_name"]].isin(selected_tags)
    ].reset_index(drop=True)

    # split data into train and test stratified by contact_reason
    train, test = train_test_split(
        merchant_data[["sentence_embeddings", params["target_name"]]],
        test_size=0.2,
        stratify=merchant_data[params["target_name"]],
        random_state=params["seed"],
    )

    return train, test, selected_tags


def preprocess_data(repo_path: Path, params: dict) -> (dict, dict):
    """
    Preprocess input data
    :param repo_path: path to repo
    :param params: dict of params
    :return: split_data
    """

    # load data from DVC specific to version
    # data_path = dvc.api.get_url(path=params['data_path'], repo=repo_path, rev=params['data_version'])
    # data = pd.read_parquet(data_path, columns=params['data_columns'])

    # load data from DVC
    data = pd.read_parquet(
        repo_path / params["data_path"], columns=params["data_columns"]
    )

    data["account_id"] = data.account_id.astype("category")
    data["email_sentence_embeddings"] = data.email_sentence_embeddings.apply(
        lambda x: json.loads(x) if x else None
    )

    # concat email sentence hashes
    data["email_sentence_hashed"] = data.email_sentence_embeddings.apply(
        lambda x: "".join([key for key in x]) if x else None
    )

    # aggregate sentence embeddings by pooling type
    if params["embedding_pooling_type"] == "mean":
        data["sentence_embeddings"] = data.email_sentence_embeddings.apply(
            lambda x: np.max([emb for _, emb in x.items()], axis=0).tolist()
            if x
            else None
        )
    else:
        data["sentence_embeddings"] = data.email_sentence_embeddings.apply(
            lambda x: np.max([emb for _, emb in x.items()], axis=0).tolist()
            if x
            else None
        )

    # preprocess data for each merchant
    train_test_dict = {}
    selected_tags_dict = {}

    for account_id in tqdm(params["account_ids"]):
        train, test, selected_tags = preprocess_data_for_merchant(
            data, account_id, params
        )
        train_test_dict[account_id] = (train, test)
        selected_tags_dict[account_id] = selected_tags

    return train_test_dict, selected_tags_dict


def save_data(repo_path: Path, train_test_dict: dict, selected_tags_dict: dict) -> None:
    """
    Save data
    :param repo_path: path to repo
    :param account_id: merchant id
    :param train_test_dict: train test split data
    :return:
    """

    # save train and test split data for each merchant
    for account_id in tqdm(train_test_dict):
        train, test = train_test_dict[account_id]
        train.to_parquet(repo_path / f"data/train_{account_id}", index=False)
        test.to_parquet(repo_path / f"data/test_{account_id}", index=False)

    # save selected tags for each merchant
    pickle.dump(selected_tags_dict, open(repo_path / "data/selected_tags.pkl", "wb"))


def main(repo_path: Path) -> None:
    """
    Main function
    :param repo_path: path to repo
    :return: None
    """

    # load params of preprocess stage
    params = dvc.api.params_show(stages="preprocess")

    # preprocess data
    train_test_dict, selected_tags_dict = preprocess_data(repo_path, params)

    # save preprocessed data
    save_data(repo_path, train_test_dict, selected_tags_dict)


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
