# murmur-classifier

## REST API 

### Setup Instructions

* Install Python 3.6.x. I would recommend using pyenv (https://github.com/pyenv/pyenv) so you can dynamically switch between Python versions.
* Create a Python virtual environment (optional) 
    * I like to use `python -m venv {ENV_NAME}` to create virtual envs
    * `source {PATH_TO_ENV}/bin/activate` to activate the env
* Install requirements by typing this command in the root directory of this project `pip install -r requirements.txt`
* Setup the matlab engine (https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html#mw_4b80d62e-4375-4310-b350-d3f4c509c83e):
    * `cd /Applications/MATLAB_R2018a.app/extern/engines/python`
        * May need to replace `/Applications/MATLAB_R2018a.app` with the right file path
    * Run `python setup.py install --prefix="/local/work/matlab18aPy36"`
* Set the following environment variables:
    * `ADABOOST_FILEPATH` --> Path to Adaboost classifier model
    * `MATLAB_SCRIPT_PATH` --> Path to directory containing MATLAB segmentation script
    * `MATLAB_ENGINE_PATH` --> "/local/work/matlab18aPy36/lib/python3.6/site-packages", or wherever your MATLAB engine is saved if not on a Mac
    * `PYTHONPATH` --> Path to murmur_classifier folder

### Running the REST API

Run the following command from the root directory of this project: `python api.py`

### Imports

Add the path to the root of this repo to your PYTHONPATH env var if you're getting import errors.
