import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
data = pd.read_excel('Dataset.xlsx')

# Data Cleaning and Preprocessing
# Convert 'Date_of_Journey' to datetime
data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], dayfirst=True)

# Extract day, month, and year from the date
data['Journey_Day'] = data['Date_of_Journey'].dt.day
data['Journey_Month'] = data['Date_of_Journey'].dt.month
data['Journey_Year'] = data['Date_of_Journey'].dt.year

# Convert 'Dep_Time' and 'Arrival_Time' to datetime and extract hours and minutes
data['Dep_Time'] = pd.to_datetime(data['Dep_Time'], format='%H:%M').dt.time
data['Dep_Hour'] = pd.to_datetime(data['Dep_Time'], format='%H:%M:%S').dt.hour
data['Dep_Minute'] = pd.to_datetime(data['Dep_Time'], format='%H:%M:%S').dt.minute

# Handle 'Arrival_Time' which sometimes has dates
def extract_arrival_time(row):
    if isinstance(row['Arrival_Time'], str):
        if ' ' in row['Arrival_Time']:
            time_part = row['Arrival_Time'].split(' ')[0]
            return pd.to_datetime(time_part, format='%H:%M').time()
    return pd.to_datetime(row['Arrival_Time'], format='%H:%M').time()

data['Arrival_Time'] = data.apply(extract_arrival_time, axis=1)
data['Arrival_Hour'] = pd.to_datetime(data['Arrival_Time'], format='%H:%M:%S').dt.hour
data['Arrival_Minute'] = pd.to_datetime(data['Arrival_Time'], format='%H:%M:%S').dt.minute

# Convert 'Duration' to total minutes
def duration_to_minutes(duration):
    if isinstance(duration, str):
        parts = duration.split()
        total_minutes = 0
        for part in parts:
            if 'h' in part:
                total_minutes += int(part.replace('h', '')) * 60
            elif 'm' in part:
                total_minutes += int(part.replace('m', ''))
        return total_minutes
    return 0

data['Duration_Minutes'] = data['Duration'].apply(duration_to_minutes)

# Encode 'Total_Stops'
stop_mapping = {
    'non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4
}
data['Total_Stops'] = data['Total_Stops'].map(stop_mapping).fillna(0)

# Feature Engineering: Extract route information
data['Route_Count'] = data['Route'].apply(lambda x: len(x.split('→')) if isinstance(x, str) else 0)

# Encode categorical variables
categorical_cols = ['Airline', 'Source', 'Destination', 'Additional_Info']
data[categorical_cols] = data[categorical_cols].astype(str)

# Preprocess categorical data using OneHotEncoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Define features and target
X = data.drop(['Price', 'Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 'Route'], axis=1)
y = data['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'R2 Score: {r2_score(y_test, y_pred)}')

# Save the model
joblib.dump(model, 'flight_fare_model.pkl')

# Example prediction
example_data = pd.DataFrame({
    'Airline': ['IndiGo'],
    'Source': ['Banglore'],
    'Destination': ['New Delhi'],
    'Total_Stops': [0],
    'Additional_Info': ['No info'],
    'Journey_Day': [24],
    'Journey_Month': [3],
    'Journey_Year': [2019],
    'Dep_Hour': [22],
    'Dep_Minute': [20],
    'Arrival_Hour': [1],
    'Arrival_Minute': [10],
    'Duration_Minutes': [170],
    'Route_Count': [2]
})

predicted_price = model.predict(example_data)
print(f'Predicted Price: {predicted_price[0]}')