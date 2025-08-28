from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# cargar modelo y columnas
model = joblib.load('modelo_regresion_lineal.pkl')
columnas = joblib.load('columnas_modelo.pkl')

app = FastAPI(title="Predicción de Precios de Casas")

class HouseData(BaseModel):
    size: float
    bedrooms: int
    age: int

@app.get("/")
def home():
    return {"message": "Bienvenido a la API de Predicción de Precios de Casas"}

@app.post("/predict")
def predict(data: HouseData):
    x_new = pd.DataFrame([[data.size, data.bedrooms, data.age]], columns=columnas)
    prediction = model.predict(x_new)
    return {"Prediccion de precio": prediction[0]}

