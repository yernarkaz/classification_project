#main.py

import json
import numpy as np
import pickle
import yaml

from fastapi import FastAPI
from utils import load_yaml_file


# deployment environment
ENV = "prod"

# load deployment config files
config = load_yaml_file("config/deployment_config.yaml")[ENV]['inference']
merchant_config = load_yaml_file("config/merchant_config.yaml")
preprocess_params = config['preprocess_params']

# load model pipelines
model_pipelines = pickle.load(open(config['model_path'], 'rb'))
tags = np.load(config['tags_path'])

app = FastAPI()

@app.post("/inference")
async def inference(account_id: str, email_sentence_embeddings: str):
    """
    Inference production model
    :param account_id:
    :param email_sentence_embeddings:
    :return contact_reason: return a contact reason tags for merchant
    """

    if account_id not in merchant_config:
        return {
            "status": "error",
            "error_message": "No merchant found"
        }

    if merchant_config[account_id] in ['ONBOARD', 'RETIRED']:
        return {
            "status": "error",
            "error_message": "Merchant is not available for inference"
        }

    sentence_embeddings = json.loads(email_sentence_embeddings)

    if preprocess_params['embedding_pooling_type'] == "mean":
        embeddings = np.mean([emb for _, emb in sentence_embeddings.items()], axis=0)
    else:
        embeddings = np.max([emb for _, emb in sentence_embeddings.items()], axis=0)
    
    model = model_pipelines[config['selected_model']]
    y_pred = model.predict([embeddings])
    
    return {
        "status": "success",
        "contact_reason": int(y_pred[0])
    }