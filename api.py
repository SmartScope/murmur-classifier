from flask import Flask, jsonify, request, abort
from flask_restplus import Resource, Api
from classification.features import FeaturesProcessor
from classification.classifier import Classifier
from classification.cnn import CNN
import pickle
import os
import sys
# Hack https://github.com/matplotlib/matplotlib/issues/13414
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wave

#  Create a Flask WSGI application
app = Flask(__name__)
#  Create a Flask-RESTPlus API
api = Api(app)

# Environment variables
adaboost_model_filepath = os.getenv('ADABOOST_FILEPATH')
cnn_model_filepath = os.getenv("CNN_FILEPATH")
matlab_engine_path = os.getenv('MATLAB_ENGINE_PATH', '/local/work/matlab18aPy36/lib/python3.6/site-packages')
matlab_script_path = os.getenv('MATLAB_SCRIPT_PATH', os.path.dirname(os.path.abspath(__file__)) + "/segmentation")

sys.path.append(matlab_engine_path)
import matlab.engine
# Instantiate matlab engine
matlab_eng = matlab.engine.start_matlab()
matlab_eng.cd(matlab_script_path, nargout=0)

# TODO: Extract this into a utils file
# Input = path to wav file
def run_adaboost_pipeline(filepath, ensemble = False):
    # Step 1: Perform segmentation using MATLAB script
    matlab_eng.segmentOneRecording(filepath, nargout=0)
    stripped_fp = filepath.split(".wav")[0]

    # Step 2: Get features
    features = get_features_from_audiofile(stripped_fp)

    # Step 3: Invoke model using features
    classifier = Classifier()
    prediction = classifier.predict(features, ensemble, adaboost_model_filepath)

    return prediction

def get_features_from_audiofile(filepath):
    features_processor = FeaturesProcessor(filepath)
    features = [features_processor.get_all_features()]
    return features

def run_cnn_pipeline(filepath, ensemble = False):
    # Step 1: Perform segmentation using MATLAB script
    matlab_eng.segmentOneRecording(filepath, nargout=0)
    stripped_fp = filepath.split(".wav")[0]

    # Step 2: Invoke model
    cnn = CNN()
    prediction = cnn.predict(stripped_fp, ensemble, cnn_model_filepath)

    return prediction

def plot_wav_file(filepath):
    # read wave file
    spf1 = wave.open(filepath, "r")

    # Extract Raw Audio from Wav File
    signal = spf1.readframes(-1)
    signal = np.fromstring(signal, "Int16")

    # FFT
    N = len(signal)

    plt.plot(np.linspace(0, N/5512.5, num=N), signal / max(abs(signal)))

    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')

    plot_fp = filepath.split(".wav")[0] + ".png"
    plt.savefig(plot_fp)
    plt.clf()
    return plot_fp

# Create a RESTful resource
@api.route('/healthcheck')
class HelloWorld(Resource):
    def get(self):
        return {'status': 'OK'}

@api.route('/classify_adaboost')
class ClassifyAdaBoost(Resource):
    def get(self):
        args = request.args
        if "filepath" not in args:
            abort(422)
        classification_result = run_adaboost_pipeline(args["filepath"])
        return jsonify({'classification': int(classification_result[0])})

@api.route('/classify_cnn')
class ClassifyCNN(Resource):
    def get(self):
        args = request.args
        if "filepath" not in args:
            abort(422)
        
        classification_result = run_cnn_pipeline(args["filepath"])
        return jsonify({
            'classification': int(classification_result)
        })

@api.route('/classify_ensemble')
class ClassifyEnsemble(Resource):
    def get(self):
        args = request.args
        if "filepath" not in args:
            abort(422)
        
        filepath = args["filepath"]
        adaboost_result = run_adaboost_pipeline(filepath, ensemble=True)[0]
        cnn_result = run_cnn_pipeline(filepath, ensemble=True)[0]

        if adaboost_result[1] > 0.6 or cnn_result[1] > 0.6:
            return jsonify({
                'classification': int(1)
            })
        else:
            return jsonify({
                'classification': int(0)
            })

@api.route('/plot_wavfile')
class ClassifyCNN(Resource):
    def get(self):
        args = request.args
        if "filepath" not in args:
            abort(422)

        plot_fp = plot_wav_file(args["filepath"])
        return jsonify({
            'plot_path': plot_fp
        })

if __name__ == '__main__':
    # Start a development server
    app.run(debug=True)