# example-model-serving

This repository contains example of serving machine learning model using FastAPI

1. You must have the model trained from repository [https://github.com/alphinside/example-wandb-trainer](https://github.com/alphinside/example-wandb-trainer) to be serve in here. The model then need to put under `model` directory.

2. To select which model to be serve, you need to edit `.env.example` to either `MODEL_TYPE=svc` or `MODEL_TYPE=decision_tree`

3. Install all dependencies

    ```shell
    poetry install
    ```

4. Run the service

    ```shell
    poetry run uvicorn example_model_serving.main:app --reload
    ```

5. Open your browser at `http://127.0.0.1:8000/docs` to try the request
