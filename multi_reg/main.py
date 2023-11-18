from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import numpy as np

class XPredict(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

app = FastAPI()

@app.get("/")
def hey():
    return "Hey"

@app.post("/predict")
def predict(predictData: XPredict):
    model_regressor = pickle.load(open('regressor.pkl', 'rb'))
    standard_scaler = pickle.load(open('scaler.pkl', 'rb'))
    input_data = [predictData.MedInc, predictData.HouseAge, predictData.AveRooms, predictData.AveBedrms,
                  predictData.Population, predictData.AveOccup, predictData.Latitude, predictData.Longitude]
    X_test_scaller = standard_scaler.transform([input_data])
    prd_data = model_regressor.predict(X_test_scaller)
    return {"data": prd_data[0]}



# from fastapi import  FastAPI
# import pickle
# from pydantic  import BaseModel
# class XPredict(BaseModel):
#     MedInc:float
#     HouseAge:float
#     AveRooms:float
#     AveBedrms:float
#     Population:float
#     AveOccup:float
#     Latitude:float
#     Longitude:float




# app=FastAPI()

# @app.get("/")
# def hey():
#     return "Hey"
# @app.post("/predict")
# def predict(predictData:XPredict):
#     model_regressor=pickle.load(open('regressor.pkl','rb'))
#     standard_scaler=pickle.load(open('scaler.pkl', 'rb'))
#     print(predictData,)
#     X_test_scaller=standard_scaler.transform(**predictData)
#     prd_data=model_regressor.predict(X_test_scaller )
#     return  {"data":prd_data }


