from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Crear una instancia de FastAPI
app = FastAPI()

# Cargar el modelo optimizado
with open("models/best_xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Definir el esquema de entrada
class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.get("/")
def home():
    return {
        "message": "Este es un modelo de clasificación para predecir la potabilidad del agua. \
                    El modelo recibe valores químicos de muestras de agua y devuelve si son potables (1) o no potables (0)."
    }

@app.post("/potabilidad/")
def predict_potability(sample: WaterSample):
    # Convertir la entrada en un arreglo numpy
    features = np.array([[sample.ph, sample.Hardness, sample.Solids, sample.Chloramines,
                          sample.Sulfate, sample.Conductivity, sample.Organic_carbon,
                          sample.Trihalomethanes, sample.Turbidity]])
    # Realizar la predicción
    prediction = model.predict(features)[0]
    
    return {"potabilidad": int(prediction)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)