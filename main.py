from fastapi import FastAPI
import pandas as pd
import joblib

# 1. Cargar modelo y columnas
modelo = joblib.load("modelo_ventas.pkl")
columnas = joblib.load("columnas.pkl")

# 2. Crear API
app = FastAPI(title="API de Predicción de Ventas")

@app.get("/")
def home():
    return {"mensaje": "Bienvenido a la api de predicción de ventas"}

@app.post("/predict")
def predict(data: dict):
    # Convertir input a DataFrame con las mismas columnas que entrenó el modelo
    X_new = pd.DataFrame([data], columns=columnas)
    #Hacer la predicción
    pred = modelo.predict(X_new)[0]
    return {"Predicción de Ventas":round(pred, 2)}