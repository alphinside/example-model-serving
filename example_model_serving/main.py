import logging
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from starlette.requests import Request

from .config import ModelType, get_settings

logging.basicConfig(level=logging.INFO)


MODEL_PATH = Path("./model")
MODEL_FILENAME = {
    ModelType.svc: "svc.joblib",
    ModelType.decision_tree: "decision_tree.joblib",
}


def load_model():
    model_fullpath = MODEL_PATH / MODEL_FILENAME[get_settings().model_type]
    model = load(model_fullpath)

    return model


app = FastAPI()


@app.on_event("startup")
def startup_events():
    logging.info("Running app startup events")
    model = load_model()
    app.state.model = model


class RequestBody(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class Response(BaseModel):
    results: List[List[float]]


@app.post("/prediction")
def post_prediction(request: Request, payload: RequestBody):
    input_df = pd.DataFrame([payload.dict()])
    input_array = input_df.to_numpy()

    model = request.app.state.model
    results = model.predict_proba(input_array)

    return Response(results=results.tolist())
