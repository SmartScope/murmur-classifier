from flask import Flask, jsonify, request, abort
from flask_restplus import Resource, Api
from classification.features import FeaturesProcessor
import pickle
import os
import sys

#  Create a Flask WSGI application
app = Flask(__name__)
#  Create a Flask-RESTPlus API
api = Api(app)

# Environment variables
adaboost_model_filepath = os.getenv('ADABOOST_FILEPATH')
matlab_engine_path = os.getenv('MATLAB_ENGINE_PATH', '/local/work/matlab18aPy36/lib/python3.6/site-packages')
matlab_script_path = os.getenv('MATLAB_SCRIPT_PATH', os.path.dirname(os.path.abspath(__file__)) + "/segmentation")

sys.path.append(matlab_engine_path)
import matlab.engine
# Instantiate matlab engine
matlab_eng = matlab.engine.start_matlab()
matlab_eng.cd(matlab_script_path, nargout=0)

# TODO: Extract this into a utils file
# Input = path to wav file
def run_adaboost_pipeline(filepath):
    # TODO Step 1: Perform segmentation using MATLAB script
    matlab_eng.segmentOneRecording(filepath, nargout=0)
    stripped_fp = filepath.split(".wav")[0]
    # Step 2: Get features
    features = get_features_from_audiofile(stripped_fp)
    # TODO Step 3: Invoke model using features
    return predict_adaboost(features)

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
        args = request.args
        if "filepath" not in args:
            abort(422)
        classification_result = run_adaboost_pipeline(args["filepath"])
        return jsonify({'classification': int(classification_result[0])})


if __name__ == '__main__':
    # Start a development server
    app.run(debug=True)