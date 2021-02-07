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

### Running the REST API

Run the following command from the root directory of this project: `python api.py`

## Electron App

### Setup instructions

* Install nodejs, npm, and yarn:
   * Note: need npm version >= 10.17.0
   * On MacOS: `brew install node npm yarn`
   * On Linux: `sudo apt install nodejs git npm` and `sudo npm install -g yarn`
* Install other system dependencies:
   * On MacOS: `brew install libnss3-dev libgdk-pixbuf2.0-dev libgtk-3-dev libxss-dev`
   * On Linux: `sudo apt install libnss3-dev libgdk-pixbuf2.0-dev libgtk-3-dev libxss-dev`
* Install Vue CLI and electron tools
   * `sudo yarn global add @vue/cli`
   * `vue add electron-builder`
* `cd my_app` and install project dependencies:
   * `yarn install`
   
### Running the App

Run the following from `my_app` directory: `yarn run electron:serve`.
   * Note: no further instructions needed on MacOS, but may need to follow these addl steps if using WSL https://www.beekeeperstudio.io/blog/building-electron-windows-ubuntu-wsl2
