import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# initialize the flask with a name
sales_forecast_api = Flask("SuperKart Predictor")

# load the trained sales forecast model
model = joblib.load("/content/drive/MyDrive/SuperKart/backend_files/sales_prediction_model_v1_0.joblib")

# define the route for the home page
@sales_forecast_api.get('/')
def home():
    return "Welcome to the Sales Forecast Prediction API!"

# define the endpoint to predict sales forecast
@sales_forecast_api.post('/v1/predict')
def predict_sales():
    # get the JSON data from the request
    predict_data = request.get_json()

    # extract relevant features from the input data.
    sample = {
        'Product_Weight': predict_data['Product_Weight'],
        'Product_Sugar_Content': predict_data['Product_Sugar_Content'],
        'Product_Allocated_Area': predict_data['Product_Allocated_Area'],
        'Product_MRP': predict_data['Product_MRP'],
        'Store_Size': predict_data['Store_Size'],
        'Store_Location_City_Type': predict_data['Store_Location_City_Type'],
        'Store_Type': predict_data['Store_Type'],
        'Store_Establishment_Year': predict_data['Store_Establishment_Year'],
        'Product_Code': predict_data['Product_Code'],
        'Product_Type': predict_data['Product_Type'],
        'Store_Id': predict_data['Store_Id']
    }

    # convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # make a sales forecast prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # return the prediction as a JSON response
    return jsonify({'Sales': prediction})

# Run the Flask app in debug mode
if __name__ == '__main__':
    sales_forecast_api.run(debug=True)
