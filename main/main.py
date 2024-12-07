import pandas as pd
from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from src.clean_data_csv import clean_data
import os
import mlflow.pyfunc

from dotenv import load_dotenv
import os
load_dotenv("../main/src/.env")

DagsHub_username = os.getenv("DagsHub_username")
DagsHub_token=os.getenv("DagsHub_token")
os.environ['MLFLOW_TRACKING_USERNAME']= DagsHub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = DagsHub_token


# setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/Znullptr/Sales-Forecast-MLOps.mlflow')  # your mlfow tracking uri

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# let's call the model from the model registry ( in production stage)

df_mlflow = mlflow.search_runs(filter_string="metrics.mse_score_test < 1")
run_id = df_mlflow.loc[df_mlflow['metrics.mse_score_test'].idxmax()]['run_id']

logged_model = f'runs:/{run_id}/ML_models'

# Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)


@app.get("/")
def read_root():
    return {"Hello": "to sales forecast app version 1"}


# this endpoint receives data in the form of csv file (histotical transactions data)
@app.post("/predict/csv")
def return_predictions(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    preprocessed_data = clean_data(data)
    predictions = model.predict(preprocessed_data)
    return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
