import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

def get_terraclimate_data(lat, lon, start_year, end_year):
    """
    Fetches TerraClimate data for a specific latitude and longitude.
    """
    print(f"Fetching data for Lat: {lat}, Lon: {lon} from {start_year} to {end_year}")
    
    # Define a point
    point = ee.Geometry.Point([lon, lat])
    
    # Load TerraClimate ImageCollection
    terraclimate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
        .filterDate(f'{start_year}-01-01', f'{end_year}-12-31') \
        .filterBounds(point)
    
    # Variables we want to extract
    bands = ['pr', 'tmmx', 'tmmn', 'soil', 'vpd', 'pet', 'pdsi', 'ro', 'def']
    
    # Check if data exists
    count = terraclimate.size().getInfo()
    if count == 0:
        print("No data found for this location and time range.")
        return pd.DataFrame()
    
    # Function to extract values for each image
    def extract_point_data(image):
        # Extract the dictionary of values
        values = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=4000
        )
        
        # Add the date
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        return ee.Feature(None, values).set('date', date)

    # Map the function over the collection
    extracted_features = terraclimate.map(extract_point_data)
    
    # Get the info and convert to pandas dataframe
    feature_info = extracted_features.getInfo()['features']
    
    data_list = []
    for f in feature_info:
        props = f['properties']
        
        # Sometimes points are masked (null data), so we handle that:
        row = {
            'date': props.get('date'),
            'lat': lat,
            'lon': lon,
            'ppt': props.get('pr'),     # Precipitation
            'tmax': props.get('tmmx'),  # Max Temp
            'tmin': props.get('tmmn'),  # Min Temp
            'soil': props.get('soil'),  # Soil Moisture
            'vpd': props.get('vpd'),    # Vapor Pressure Deficit
            'pet': props.get('pet'),    # Potential Evapotranspiration
            'pdsi': props.get('pdsi'),  # Palmer Drought Severity Index
            'ro': props.get('ro'),      # Runoff
            'def': props.get('def')     # Climatic Water Deficit
        }
        data_list.append(row)
        
    df = pd.DataFrame(data_list)
    
    # Convert 'date' to datetime
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        
        # The TerraClimate dataset has scale factors. We need to apply them.
        # pr: no scale factor needed (already mm) but let's check TerraClimate docs: 
        # Actually tmmx/tmmn has a scale factor of 0.1, vpd has 0.01, pdsi has 0.01.
        if 'tmax' in df.columns: df['tmax'] = df['tmax'] * 0.1
        if 'tmin' in df.columns: df['tmin'] = df['tmin'] * 0.1
        if 'vpd' in df.columns: df['vpd'] = df['vpd'] * 0.01
        if 'pdsi' in df.columns: df['pdsi'] = df['pdsi'] * 0.01
        
    return df

if __name__ == "__main__":
    # Test coordinates (e.g., California)
    test_lat = 36.7783
    test_lon = -119.4179
    # Time range from 1958 to 2022
    df = get_terraclimate_data(test_lat, test_lon, '1958', '2022')
    
    if not df.empty:
        # Save to CSV
        output_file = 'terraclimate_data.csv'
        df.to_csv(output_file, index=False)
        print(f"Data successfully extracted and saved to {output_file}")
        print(df.head())
    else:
        print("Data extraction failed.")
