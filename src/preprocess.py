# preprocess.py

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import dvc.api
import json
import pandas as pd
import numpy as np


def preprocess_data(params):
    """
    TODO: add description
    """

    data = pd.read_parquet(params["data_path"], columns=params['data_columns'])
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
        merchant_data.mean_pooled_embeddings, 
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
    TODO: add description
    """
    train_list, test_list = [], []
    for tag in tqdm(selected_tags):
        train, test = split_data[tag]
        train_list.append(train)
        test_list.append(test)

    pd.concat(train_list).reset_index(drop=True).to_parquet(repo_path + "/data/train")
    pd.concat(test_list).reset_index(drop=True).to_parquet(repo_path + "/data/test")
    
    np.save(selected_tags, repo_path + "/data/selected_tags.npy")


def main(repo_path):
    """
    TODO: add description
    """

    print("repo_path:", repo_path)
    params = dvc.api.params_show()
    print("params:")
    print(params)

    print("Preprocessing data...")
    split_data, selected_tags = preprocess_data(params)
    print("Saving data...")
    save_data(repo_path, split_data, selected_tags)


if __name__ == "__main__":
    repo_path = Path(__file__).parent
    main(repo_path)
