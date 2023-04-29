#!/usr/bin/python
from flask import Flask
#from flask_restplus import Api, Resource, fields, reqparse
from flask_restx import Api, Resource, fields, reqparse
import joblib
#from model_deployment.m09_model_deployment import predicted_price
#from m09_model_deployment import predicted_price 
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
    'vehicle_data', 
    type=str, 
    required=True, 
 #   help='Vehicle data in JSON format with the following fields: Price, Year, Mileage, State, Make, Model', 
    help='Vehicle data in JSON format with the following fields: Year, Mileage, State, Make, Model', 
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

        # Convert JSON data to DataFrame
        vehicle_data = pd.read_json(args['vehicle_data'], orient='records')

        # Apply one-hot encoding to State, Make, and Model
        one_hot_data = pd.get_dummies(vehicle_data, columns=['State', 'Make', 'Model'])

        # Make prediction
        price_prediction = predicted_price(one_hot_data)

        return {
         "result": price_prediction
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

