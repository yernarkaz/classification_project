#main.py

import json

from fastapi import FastAPI

app = FastAPI()

@app.post("/inference")
async def inference(account_id: str, email_sentence_embeddings: str):
    """
    Inference production model
    :param account_id:
    :param email_sentence_embeddings:
    :return contact_reason: return a contact reason f
    """
    print("Inferencing for account :", account_id)
    print("Email sentence embeddings :", json.dumps(email_sentence_embeddings))

    return {"contact_reason": "--------"}