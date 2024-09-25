from fastapi import Depends, HTTPException, Header
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)

# Path to the trained model
model_path = 'models/trained_model.pkl'
api_key = 'your_secret_api_key'

app = FastAPI()

# Load the trained model with error handling
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logging.info("Model loaded successfully from %s", model_path)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

class PropertyInfo(BaseModel):
    """
    PropertyInfo defines the input data structure for property valuation.
    
    Fields:
    - type: str: Type of property (e.g., 'house', 'apartment').
    - sector: str: Geographical sector of the property.
    - net_usable_area: float: Usable area of the property in square meters.
    - net_area: float: Total area of the property in square meters.
    - n_rooms: float: Number of rooms in the property.
    - n_bathroom: float: Number of bathrooms in the property.
    - latitude: float: Latitude of the property location.
    - longitude: float: Longitude of the property location.
    - price: int: Actual price of the property.
    """
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
    """
    Verifies the API key provided in the request headers.
    
    Parameters:
    - x_api_key: str: API key passed in the request header.
    
    Raises:
    - HTTPException: If the API key is invalid or missing.
    """
    if x_api_key != api_key:
        logging.warning("Invalid API key attempt.")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    logging.info("API key verified successfully.")


@app.get("/")
async def read_root():
    """
    Health check endpoint to verify if the API is running.
    
    Returns:
    - dict: A simple JSON response confirming the API is running.
    """
    logging.info("Health check request received.")
    return {"health_check": "OK", "model_version": 1}


@app.post("/predict/", dependencies=[Depends(verify_api_key)])
def predict_price(property_info: PropertyInfo):
    """
    Predicts the estimated property price using the trained model.
    
    Parameters:
    - property_info: PropertyInfo: Input property details.

    Returns:
    - dict: A dictionary containing the estimated price of the property.
    
    Raises:
    - HTTPException: If prediction fails due to data errors or model issues.
    """
    try:
        # Prepare the input data as a DataFrame for the model
        data = pd.DataFrame([{
            "type": property_info.type, 
            "sector": property_info.sector, 
            "net_usable_area": property_info.net_usable_area, 
            "net_area": property_info.net_area,   
            "n_rooms": property_info.n_rooms, 
            "n_bathroom": property_info.n_bathroom, 
            "latitude": property_info.latitude, 
            "longitude": property_info.longitude,
            "price": property_info.price                                 
        }])
        
        logging.info(f"Received prediction request for data: {data}")

        # Make prediction using the model pipeline
        prediction = model.predict(data)

        logging.info(f"Prediction made successfully: {prediction[0]}")
        
        # Return the predicted property price
        return {"estimated_price": prediction[0]}
    
    except ValueError as e:
        logging.error(f"Data error during prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again later.")


if __name__ == "__main__":
    import uvicorn
    logging.info("Starting API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
