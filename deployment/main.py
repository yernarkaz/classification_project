# main.py

import json
import numpy as np
import pickle
import timeit

from fastapi import FastAPI
from utils import load_yaml_file


# deployment environment
ENV = "prod"

# load deployment config files
config = load_yaml_file("config/deployment_config.yaml")[ENV]["inference"]
merchant_config = load_yaml_file("config/merchant_config.yaml")
model_params = config["selected_model"]

# load model pipelines
model_pipelines = pickle.load(open(config["model_path"], "rb"))

app = FastAPI()


@app.post("/inference")
async def inference(account_id: str, email_sentence_embeddings: str) -> dict:
    """
    Inference production model
    :param account_id:
    :param email_sentence_embeddings:
    :return contact_reason: return a contact reason tags for merchant
    """
    start = timeit.default_timer()

    if account_id not in merchant_config:
        return {"status": "error", "error_message": "No merchant found"}

    if merchant_config[account_id] in ["ONBOARD", "RETIRED"]:
        return {
            "status": "error",
            "error_message": "Merchant is not available for inference",
        }

    # get model pipeline for merchant
    models = model_pipelines[account_id]

    # parse email sentence embeddings
    sentence_embeddings = json.loads(email_sentence_embeddings)

    # aggregate sentence embeddings by pooling type
    if model_params["embedding_pooling_type"] == "mean":
        embeddings = np.mean([emb for _, emb in sentence_embeddings.items()], axis=0)
    else:
        embeddings = np.max([emb for _, emb in sentence_embeddings.items()], axis=0)

    model_pipeline = models[model_params["name"]]
    contact_reason = model_pipeline.predict([embeddings])[0]

    stop = timeit.default_timer()
    return {
        "status": "success",
        "contact_reason": contact_reason.replace('"', ""),
        "inference_time": "%.1f ms" % (1000 * (stop - start)),
    }
