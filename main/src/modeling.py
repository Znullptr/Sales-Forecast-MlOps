#from dotenv import load_dotenv
import os
from data_preprocessing_training import preprocess_data, transform_data
import pandas as pd
from imblearn.over_sampling import SMOTE
import mlflow
from treatement import VARModel
from treatement import XGBoostModel

#Load environement variable (Dagshub credentials)
# from dotenv import load_dotenv
# import os
# load_dotenv("../backend/src/.env")

# DagsHub_username = os.getenv("DagsHub_username")
# DagsHub_token=os.getenv("DagsHub_token")

#Get Dagshub credentials
# os.environ['MLFLOW_TRACKING_USERNAME']= DagsHub_username
# os.environ["MLFLOW_TRACKING_PASSWORD"] = DagsHub_token

#Affect Daghsub credentials

os.environ['MLFLOW_TRACKING_USERNAME']= "..."
os.environ["MLFLOW_TRACKING_PASSWORD"] = "..."

#setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/.../....mlflow') #your mlfow tracking uri
mlflow.set_experiment("sales-forecast-experiment")

#Data Url and version
version = "v2.0"
data_url = "../../data/Cleaned_Sales_Data.xlsx"

#read the data
df = pd.read_csv(data_url)
#cleaning and preprocessing
product_dataframes = transform_data(df)

for df_product in product_dataframes:
    X_train,X_test,y_train,y_test = preprocess_data(df_product)

    #Execute the models with new version of data:
    VARModel(data_url,version,df,X_train,X_test)
    XGBoostModel(data_url,version,df,X_train,y_train,X_test,y_test)

