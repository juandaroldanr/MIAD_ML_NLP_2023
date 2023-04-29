#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def preprocess_data(data):
    # Transform State, Make, and Model into dummy variables
    data = pd.get_dummies(data, columns=['State'])
    data = pd.get_dummies(data, columns=['Make'])
    data = pd.get_dummies(data, columns=['Model'])

    # Drop the Price column from the training set
    if 'Price' in data.columns:
        data.drop('Price', axis=1, inplace=True)

    return data


def predict_price(year, mileage, state, make, model):
    # Load the saved model
    clf = joblib.load(os.path.dirname(__file__) + '/car_price_prediction_clf.pkl')

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Year': [year],
        'Mileage': [mileage],
        'State_' + state: [1],
        'Make_' + make: [1],
        'Model_' + model: [1]
    })

    # Apply the same preprocessing as in the training and testing data
    input_data = preprocess_data(input_data)

    # Make prediction
    predicted_price = clf.predict(input_data)[0]

    return predicted_price


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print('Please add values for Year, Mileage, State, Make, and Model')
    else:
        year = int(sys.argv[1])
        mileage = int(sys.argv[2])
        state = sys.argv[3]
        make = sys.argv[4]
        model = sys.argv[5]

        predicted_price = predict_price(year, mileage, state, make, model)

        print('Predicted price: $', predicted_price)

        