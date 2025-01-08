import requests
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import time
import geopandas as gpd
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions, BBox

# Sentinel Hub configuration (Replace with your Sentinel Hub credentials)
config = SHConfig()
config.instance_id = 'YOUR_INSTANCE_ID'
config.sh_client_id = 'YOUR_CLIENT_ID'
config.sh_client_secret = 'YOUR_CLIENT_SECRET'

# Function to retrieve ship data from MarineTraffic API
def get_all_ships_data(api_key):
    url = f"https://services.marinetraffic.com/api/exportvessels/v:2/{api_key}/protocol:jsono"
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data)

# Function to get Sentinel data for a specific bounding box
def get_sentinel_data(bbox_coords, time_interval):
    bbox = BBox(bbox=bbox_coords, crs='EPSG:4326')
    bbox_size = bbox_to_dimensions(bbox, resolution=10)
    
    request = SentinelHubRequest(
        evalscript="""
        Basic script to fetch true color images
        More advanced scripts can be used to detect ships, anomalies, etc.
        Change evalscript as needed""",
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=time_interval,
                mosaicking_order='mostRecent'
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=bbox,
        size=bbox_size,
        config=config
    )
    
    response = request.get_data()
    return response[0]

# Function to detect anomalies in ship movement
def detect_anomalies(data):
    # Extract relevant features for anomaly detection
    features = data[['LAT', 'LON', 'SPEED', 'COURSE']]
    
    # Replace missing values with the mean of the column
    features.fillna(features.mean(), inplace=True)
    
    # Use Isolation Forest for anomaly detection
    model = IsolationForest(contamination=0.1)
    data['anomaly'] = model.fit_predict(features)
    
    # -1 indicates anomaly
    anomalies = data[data['anomaly'] == -1]
    return anomalies

# Main monitoring function
def monitor_ships(api_key, interval=600, satellite_interval="2024-01-01T00:00:00/2024-01-31T23:59:59"):
    while True:
        try:
            # Step 1: Get data for all ships
            ship_data = get_all_ships_data(api_key)
            
            # Step 2: Detect anomalies in ship data
            anomalies = detect_anomalies(ship_data)
            
            # Step 3: Integrate Sentinel data
            if not anomalies.empty:
                print("Anomalies detected:")
                print(anomalies)
                
                # Get Sentinel data for the bounding box around anomalies
                for index, row in anomalies.iterrows():
                    bbox_coords = [row['LON']-0.1, row['LAT']-0.1, row['LON']+0.1, row['LAT']+0.1]
                    sentinel_image = get_sentinel_data(bbox_coords, satellite_interval)
                    
                    # Process sentinel_image to detect further anomalies or verify AIS data
                    # This part would involve more advanced image processing techniques
                    
                    print(f"Processed Sentinel data for anomaly at: {row['LAT']}, {row['LON']}")
                    
            else:
                print("No anomalies detected at this time.")
            
            # Wait for the specified interval before the next fetch
            time.sleep(interval)
        
        except Exception as e:
            print("Error fetching or processing data:", e)
            # Wait before trying again to avoid rapid failure loop
            time.sleep(interval)

# Example Usage
api_key = "YOUR_API_KEY"  # Replace with your actual MarineTraffic API key
monitor_ships(api_key)
