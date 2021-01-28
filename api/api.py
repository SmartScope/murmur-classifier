from flask import Flask
from flask_restplus import Resource, Api

#  Create a Flask WSGI application
app = Flask(__name__)
#  Create a Flask-RESTPlus API
api = Api(app)


# Create a RESTful resource
@api.route('/healthcheck')
class HelloWorld(Resource):
    def get(self):
        return {'status': 'OK'}


if __name__ == '__main__':
    # Start a development server
    app.run(debug=True)