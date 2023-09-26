# preprocess.py
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import dvc.api
import json
import pandas as pd
import numpy as np


def preprocess_data(repo_path, params):
    """
    Preprocess input data
    :param repo_path: path to repo
    :param params: dict of params
    :return: split_data
    """

    data_path = dvc.api.get_url(path=params['data_path'], repo=repo_path, rev=params['data_version'])
    data = pd.read_parquet(data_path, columns=params['data_columns'])

    data["account_id"] = data.account_id.astype("category")
    data["email_sentence_embeddings"] = data.email_sentence_embeddings.apply(lambda x: json.loads(x) if x else None)

    if params["embedding_pooling_type"] == "mean":
        data["sentence_embeddings"] = data.email_sentence_embeddings.apply(lambda x: np.max([emb for _, emb in x.items()], axis=0).tolist()
                                                                            if x else None)
    else:
        data["sentence_embeddings"] = data.email_sentence_embeddings.apply(lambda x: np.max([emb for _, emb in x.items()], axis=0).tolist()
                                                                            if x else None)

    merchant_data = data[data.account_id.isin(params["account_ids"])].dropna(subset="sentence_embeddings").reset_index(drop=True)
    clean_data = pd.concat([
        merchant_data.sentence_embeddings, 
        pd.get_dummies(merchant_data[params["target_label"]]).astype(int)], 
    axis=1).reset_index(drop=True)

    tags = clean_data.columns[1:].tolist()
    target = clean_data[tags]
    target_cnt = target.sum(axis=0)
    selected_tags = list(set(tags) - set(target_cnt[target_cnt == 1].index.tolist()))
    
    split_data = {}
    for tag in tqdm(selected_tags):
        train, test = train_test_split(clean_data[["sentence_embeddings", tag]], test_size=0.2, 
                                        stratify=target[tag], shuffle=True, random_state=params["seed"])
        split_data[tag] = (train, test)

    return split_data, selected_tags


def save_data(repo_path, split_data, selected_tags):
    """
    Save data
    :param repo_path: path to repo
    :param split_data: data containing split of train and test
    :param selected_tags: tags of interest
    :return:
    """

    for tag in tqdm(selected_tags):
        train, test = split_data[tag]
        tag = tag.replace('/', '_')
        train.to_parquet(repo_path / f"data/train/{tag}", index=False)
        test.to_parquet(repo_path / f"data/test/{tag}", index=False)
    
    np.save(repo_path / "data/tags.npy", selected_tags)


def main(repo_path):
    """
    Main function
    :param repo_path: path to repo
    :return: None
    """

    print("repo_path:", repo_path)
    params = dvc.api.params_show(stages="preprocess")
    print("params:")
    print(params)

    print("Preprocessing data...")
    split_data, selected_tags = preprocess_data(repo_path, params)
    print("Saving data...")
    save_data(repo_path, split_data, selected_tags)


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
