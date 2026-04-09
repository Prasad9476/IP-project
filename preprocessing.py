import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

def create_drought_classes(pdsi):
    if pdsi >= -0.5: return 0  # No Drought
    elif -1.0 <= pdsi < -0.5: return 1  # Mild
    elif -2.0 <= pdsi < -1.0: return 2  # Moderate
    elif -3.0 <= pdsi < -2.0: return 3  # Severe
    else: return 4  # Extreme Drought

def preprocess_data(csv_file, save_scaler_path='scaler.pkl'):
    df = pd.read_csv(csv_file, parse_dates=['date'])
    df = df.sort_values(by='date')
    
    # Handle missing values using forward fill for time-series
    df = df.ffill()
    
    # Target Encoding
    df['drought_class'] = df['pdsi'].apply(create_drought_classes)
    
    # Cyclical encoding for month
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Create Lag features (1, 3, 6, 12 months)
    for lag in [1, 3, 6, 12]:
        df[f'ppt_lag_{lag}'] = df['ppt'].shift(lag)
        df[f'soil_lag_{lag}'] = df['soil'].shift(lag)
        
    # Rolling Means
    df['ppt_roll_3'] = df['ppt'].rolling(window=3).mean()
    df['ppt_roll_6'] = df['ppt'].rolling(window=6).mean()
    
    # Simple SPI (Standardized Precipitation Index) approximation 
    df['spi_3'] = (df['ppt_roll_3'] - df['ppt_roll_3'].mean()) / df['ppt_roll_3'].std()
    
    # Drop rows with NaNs caused by lags and rolling windows
    df = df.dropna()
    
    # Features to use and normalize
    features_to_normalize = [
        'ppt', 'tmax', 'tmin', 'soil', 'vpd', 'pet', 'ro', 'def', 
        'ppt_lag_1', 'soil_lag_1', 'ppt_lag_3', 'soil_lag_3', 
        'ppt_lag_6', 'soil_lag_6', 'ppt_lag_12', 'soil_lag_12',
        'ppt_roll_3', 'ppt_roll_6', 'spi_3'
    ]
                             
    scaler = MinMaxScaler()
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    
    # Save the scaler for real-time inference
    joblib.dump(scaler, save_scaler_path)
    
    # Split
    X = df[features_to_normalize + ['month_sin', 'month_cos']]
    y = df['drought_class']
    
    # Time-series Split: 70% Train, 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, df

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, df = preprocess_data('terraclimate_data.csv')
    print("Preprocessing successful.")
    print("Train shape:", X_train.shape, "Val shape:", X_val.shape, "Test shape:", X_test.shape)
