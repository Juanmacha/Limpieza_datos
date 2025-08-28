from fastapi import FastAPI
import pandas as pd
import joblib

# 1. cargar el modelo y las columnas
model = joblib.load('modelo_regresion_lineal.pkl')
columnas = joblib.load('columnas_modelo.pkl')

app = FastAPI(title="Predicción de Precios de Casas")

@app.get("/")
def home():
    return {"message": "Bienvenido a la API de Predicción de Precios de Casas"}

@app.post("/predict")
def predict(size: float, bedrooms: int, age: int):
    x_new = pd.DataFrame([[size, bedrooms, age]], columns=columnas)
    prediction = model.predict(x_new)
    return {"predicted_price": prediction[0]}
