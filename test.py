import joblib

try:
    model = joblib.load("flight_price_model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
