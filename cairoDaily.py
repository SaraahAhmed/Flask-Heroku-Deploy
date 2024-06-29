from flask import Flask, jsonify
import requests
from datetime import datetime
from itertools import groupby
import numpy as np
import pandas as pd
import joblib

# Create a Flask app
app = Flask(__name__)

# Function to get the weather forecast and energy prediction
def get_weather_forecast():
    # API key for accessing OpenWeatherMap API
    api_key = "7eb6961620c5f73cfe0df6b720888be1"

    # Coordinates (latitude and longitude) of the location you're interested in
    latitude = 30.0444
    longitude = 31.2357

    # API endpoint for hourly weather forecast data
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"

    # Make a GET request to the API
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Convert the response to JSON format
        data = response.json()

        # Load the model
        model = joblib.load(r"Daily\cairoDaily\cairoDaily_model_svr.joblib")
        
        # Load the scalers and cyclical transformer
        scaler = joblib.load(r'Daily\cairoDaily\cairoDaily_scaler_svr.pkl')
        cyclical = joblib.load(r'Daily\cairoDaily\cairoDaily_cyclical_svr.pkl')

        # Extract hourly forecast data for the next 5 days
        forecast_data = data["list"]

        # Initialize lists to store hourly and daily weather data
        hourly_list = []
        daily_list = []

        # Iterate through the hourly forecast data
        for hour in forecast_data:
            # Extract date and time for the forecast
            date_time = datetime.strptime(hour["dt_txt"], '%Y-%m-%d %H:%M:%S')

            # Add hourly data to the hourly list
            hourly_list.append({
                "date": date_time,
                "rain": hour["rain"]["3h"] if "rain" in hour else 0,
                "temp": hour["main"]["temp"] if "main" in hour else 0,
                "humidity": hour["main"]["humidity"] if "main" in hour else 0,
                "pressure": hour["main"]["pressure"] / 10 if "main" in hour else 0,
                "wind_speed": hour["wind"]["speed"] if "wind" in hour else 0
            })

        # Aggregate hourly data into daily averages or summaries
        grouped_by_date = groupby(hourly_list, lambda x: x["date"].date())
        for date, group in grouped_by_date:
            # Calculate daily averages or summaries
            group_list = list(group)
            daily_average_temp = sum(item["temp"] for item in group_list) / len(group_list)
            daily_max_rain = max(item["rain"] for item in group_list)
            daily_max_wind_speed = max(item["wind_speed"] for item in group_list)
            daily_average_humidity = sum(item["humidity"] for item in group_list) / len(group_list)
            daily_average_pressure = sum(item["pressure"] for item in group_list) / len(group_list)

            # Add daily data to the daily list
            daily_list.append({
                "date": date.strftime('%Y-%m-%d'),
                "average_temp": daily_average_temp,
                "average_humidity": daily_average_humidity,
                "max_rain": daily_max_rain,
                "average_pressure": daily_average_pressure,
                "max_wind_speed": daily_max_wind_speed,
                "PredictedAllSky": 0
            })

            # Split date to year, month, day
            year = date.year
            month = date.month
            day = date.day
            last_item = daily_list[-1]
            
            # New data
            new_data = pd.DataFrame([[year, month, day, last_item["average_temp"], last_item["average_humidity"],
                                      last_item["max_rain"], last_item["average_pressure"], last_item["max_wind_speed"]]],
                                    columns=["YEAR", "MO", "DY", "T2M", "RH2M", "PRECTOTCORR", "PS", "WS10M"])
            
            # Apply cyclical transformation to the first 3 attributes
            new_data_cyclical = cyclical.transform(new_data[['YEAR', 'MO', 'DY']])
            
            # Apply scaling to the rest of the attributes
            new_data_scaled = scaler.transform(new_data[["T2M", "RH2M", "PRECTOTCORR", "PS", "WS10M"]])
            
            # Concatenate the transformed data
            new_data_transformed = np.concatenate((new_data_cyclical, new_data_scaled), axis=1)
            
            # Predict
            prediction = model.predict(new_data_transformed)
            last_item["PredictedAllSky"] = float(prediction[0])
        
        return daily_list

# Define an API endpoint
@app.route('/api/Cairo_daily', methods=['GET'])
def daily_weather_forecast():
    data = get_weather_forecast()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
