#!/usr/bin/python
from flask import Flask, request
from flask_restx import Api, Resource, fields, reqparse
import joblib
from m09_model_deployment import predict_price
import pandas as pd

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Vehicle Price Prediction API',
    description='Vehicle Price Prediction API')

ns = api.namespace('vehicle_predict', 
     description='Vehicle Price Prediction')
   
parser = reqparse.RequestParser()

parser.add_argument(
    'year', 
    type=int, 
    required=True, 
    help='Year of the vehicle', 
    location='args')

parser.add_argument(
    'mileage', 
    type=int, 
    required=True, 
    help='Mileage of the vehicle', 
    location='args')

parser.add_argument(
    'state', 
    type=str, 
    required=True, 
    help='State where the vehicle is located', 
    location='args')

parser.add_argument(
    'make', 
    type=str, 
    required=True, 
    help='Make of the vehicle', 
    location='args')

parser.add_argument(
    'model', 
    type=str, 
    required=True, 
    help='Model of the vehicle', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.Float,
})

@ns.route('/')
class VehiclePriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()

        # Create a dictionary with the request arguments
        vehicle_data = {
            'Year': args['year'],
            'Mileage': args['mileage'],
            'State': args['state'],
            'Make': args['make'],
            'Model': args['model']
        }

        # Create a DataFrame from the dictionary
        vehicle_data_df = pd.DataFrame(vehicle_data, index=[0])

        # Apply one-hot encoding to State, Make, and Model
        one_hot_data = pd.get_dummies(vehicle_data_df, columns=['State', 'Make', 'Model'])

        # Make prediction
        price_prediction = predict_price(one_hot_data)

        return {
            "result": price_prediction
        }, 200
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

