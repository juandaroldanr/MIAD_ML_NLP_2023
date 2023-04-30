
from flask import Flask
from flask import jsonify
from flask import request
import json

import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

def load_preprocessor():
    ohe = joblib.load('preprocessor_ohe.pkl')
    return ohe

def preprocessing(raw_data, ohe):
    encoded = ohe.transform(raw_data[['State','Make','Model']]).toarray()

    df_encoded_columns = pd.DataFrame(columns=ohe.get_feature_names_out(['State','Make','Model']),data=encoded)
    df = raw_data.drop(columns=['State','Make','Model']).reset_index(drop=True)
    data_prep = pd.concat([df,df_encoded_columns],axis=1)

    return data_prep

def execute_prediction(data_preprocessed):
    return model.predict(data_preprocessed)

def make_input_df(year,miles,state,make,model):
    df = pd.DataFrame(data={'Year':[year],'Mileage':[miles],'State':[state],'Make':[make],'Model':[model]})
    return df


@app.route('/search/', methods=['POST'])
def perform_comparison():
    args = request.args.to_dict()
    
    year = int(args['year'])
    miles = float(args['miles'])
    state = str(args['states'])
    make = str(args['make'])
    model = str(args['model'])

    input_df = make_input_df(year,miles,state,make,model)
    input_prep = preprocessing(input_df,ohe)

    result = execute_prediction(input_prep)
    return json.dumps(result[0])

if __name__ == '__main__':
    print('Starting Cars API')

    model = joblib.load('model.pkl')
    print('Loaded model')

    ohe = load_preprocessor()
    print('Loaded preprocessor')

    app.run(debug=True, host = 18.118.211.25, port = 5000)
