from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, db

# Define expected input schema
class FlightInput(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Total_Stops: int
    Additional_Info: str
    Journey_Day: int
    Journey_Month: int
    Journey_Year: int
    Dep_Hour: int
    Dep_Minute: int
    Arrival_Hour: int
    Arrival_Minute: int
    Duration_Minutes: int
    Route_Count: int

# Initialize FastAPI
app = FastAPI()

# 🔹 Initialize Firebase
cred = credentials.Certificate("cost-prediction-f3321-firebase-adminsdk-fbsvc-49ecf886e7.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://cost-prediction-f3321-default-rtdb.firebaseio.com/'
})
ref = db.reference("/predictions")

# 🔹 Load trained pipeline model
model = joblib.load("flight_fare_model.pkl")

@app.get("/")
def home():
    return {"message": "Flight Price Prediction API is running!"}

@app.post("/predict")
def predict(data: FlightInput):
    try:
        # Convert to DataFrame with matching column names
        input_df = pd.DataFrame([data.dict()])
        print("Received input data:\n", input_df)

        # Make prediction using the full pipeline
        prediction = model.predict(input_df)[0]

        # Store result in Firebase
        result = {
            "input": data.dict(),
            "predicted_price": round(prediction, 2)
        }
        ref.push(result)

        return {"predicted_price": round(prediction, 2)}

    except Exception as e:
        return {"error": str(e)}
