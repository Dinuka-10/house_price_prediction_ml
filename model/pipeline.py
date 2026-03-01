import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def preprocess_input(raw_data):
    """
    Cleans data and engineers the 'house_age' feature to match training.
    """
    df = pd.DataFrame(raw_data)
    
    df['house_age'] = 2015 - df['yr_built']
    
   
    cols_to_drop = ['id', 'date', 'zipcode', 'yr_renovated']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], axis=1)
    
    return df

def predict_house_price(input_features):
    
    try:
        model = joblib.load('house_price_model.pkl')
    except FileNotFoundError:
        return "Error: house_price_model.pkl not found in current directory."
    
    
    processed_data = preprocess_input(input_features)
    

    expected_columns = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
        'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
        'sqft_basement', 'yr_built', 'lat', 'long', 'sqft_living15', 
        'sqft_lot15', 'house_age'
    ]
    processed_data = processed_data[expected_columns]
    
    log_prediction = model.predict(processed_data)
    
    actual_price = np.expm1(log_prediction)
    
    return actual_price[0]

if __name__ == "__main__":
    
    new_house = {
        'bedrooms': [3],
        'bathrooms': [2.25],
        'sqft_living': [2570],
        'sqft_lot': [7242],
        'floors': [2.0],
        'waterfront': [0],
        'view': [0],
        'condition': [3],
        'grade': [7],
        'sqft_above': [2170],
        'sqft_basement': [400],
        'yr_built': [1951],
        'lat': [47.7210],
        'long': [-122.319],
        'sqft_living15': [1690],
        'sqft_lot15': [7639]
    }

    print("Running prediction...")
    result = predict_house_price(new_house)
    
    if isinstance(result, str):
        print(result)
    else:
        print("-" * 30)
        print(f"Prediction successful!")
        print(f"Estimated House Price: ${result:,.2f}")
        print("-" * 30)