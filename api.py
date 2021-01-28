from flask import Flask
from flask_restplus import Resource, Api
from classification.features import FeaturesProcessor
import pickle
import os

#  Create a Flask WSGI application
app = Flask(__name__)
#  Create a Flask-RESTPlus API
api = Api(app)

# Environment variables
adaboost_model_filepath = os.getenv('ADABOOST_FILEPATH')

# TODO: Extract this into a utils file
def get_features_from_audiofile(filepath):
    features_processor = FeaturesProcessor(filepath)
    features = [features_processor.get_all_features()]
    return features

# Make a prediction using our Adaboost classifier
def predict_adaboost(features):
    # Load the model from disk
    loaded_model = pickle.load(open(adaboost_model_filepath, 'rb'))
    return loaded_model.predict(features)

# Create a RESTful resource
@api.route('/healthcheck')
class HelloWorld(Resource):
    def get(self):
        return {'status': 'OK'}

@api.route('/classify')
class Classify(Resource):
    def get(self):
        return get_features_from_audiofile()


if __name__ == '__main__':
    # Start a development server
    app.run(debug=True)