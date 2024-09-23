from fastapi import Depends, HTTPException, Header
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

model_path = 'models/trained_model.pkl'
api_key='your_secret_api_key'

app = FastAPI()

# Load the trained model
model = joblib.load(model_path)

class PropertyInfo(BaseModel):
    type: str   
    sector: str    
    net_usable_area: float
    net_area: float
    n_rooms: float
    n_bathroom: float
    latitude: float
    longitude: float
    price: int


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}

@app.post("/predict/", dependencies=[Depends(verify_api_key)])
def predict_price(property_info: PropertyInfo):
    # Prepare the input data as a DataFrame for the model
    data = pd.DataFrame([{
        "type": property_info.type, 
        "sector": property_info.sector, 
        "net_usable_area": property_info.net_usable_area, 
        "net_area": property_info.net_area,   
        "n_rooms": property_info.n_rooms, 
        "n_bathroom ": property_info.n_bathroom, 
        "latitude": property_info.latitude, 
        "longitude": property_info.longitude,
        "price": property_info.price                                 
    }])
    
    # Make prediction using the model pipeline
    prediction = model.predict(data)

    # Return the predicted property price
    return {"estimated_price": prediction[0]}

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)    
    