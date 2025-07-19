from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load('models/cltv_model.pkl')

class CLTVInput(BaseModel):
    Recency: int
    Frequency: int
    First_Purchase: int

@app.post("/predict_cltv")
def predict_cltv(data: CLTVInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"Predicted_CLTV": prediction}
