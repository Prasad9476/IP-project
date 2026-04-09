import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_mock_model(model_path='models'):
    os.makedirs(model_path, exist_ok=True)
    # Train dummy XGBoost for features
    X_dummy = np.random.rand(100, 20)
    y_dummy = np.random.randint(0, 5, 100)
    
    xgb = XGBClassifier()
    xgb.fit(X_dummy, y_dummy)
    joblib.dump(xgb, f'{model_path}/xgboost_model.pkl')
    
    # Simple Mock LSTM model
    model = Sequential([
        LSTM(64, input_shape=(1, 20), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(5, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save(f'{model_path}/lstm_model.h5')
    print("Mock Models generated successfully for local testing!")

if __name__ == "__main__":
    if not os.path.exists('terraclimate_data.csv'):
        print("Real data not found. Creating mock models to unblock UI...")
        create_mock_model()
    else:
        print("Data found! Training real models will go here.")
        create_mock_model() # Using mock models for this local preview
