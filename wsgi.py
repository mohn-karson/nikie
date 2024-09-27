import os
import time
import logging
import random
from glob import glob
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from config.config import config
from predict import Predictor
from scripts.utils import service_manager_loader


def allowed_file(filename):
    return all(
        ['.' in filename, filename.split('.', 1)[-1].lower() in config['DEFAULTS']['ALLOWED-FORMATS']]
    )


def delete(file_path):

    if os.path.exists(file_path):
        for file in glob(file_path+'/*'):
            if not os.path.isdir(file):
                os.remove(file)
            else:
                delete(file)
        os.rmdir(file_path)


service_manager = service_manager_loader()
logger_path = service_manager['LOGGER']['PATH']

logging.basicConfig(
    # format=service_manager['LOGGER']['FORMAT'],
    format='%(levelname)s: %(asctime)s - %(message)s',
    filename=logger_path,
    level=logging.INFO
)

application = Flask(__name__)
CORS(application)
logging.info('Service Restarted Successfully!')
predictor = Predictor()


@application.route('/', methods=['GET', 'POST'])
def home():
    return 'Service is Up and Running.'


@application.route('/get-nutrition-details', methods=['GET', 'POST'])
def entity_tagging():

    if request.method == 'POST':

        data = request.files.getlist('images')

        data = [file for file in data if allowed_file(secure_filename(file.filename))]

        if data:
            _base_path = os.path.dirname(__file__)

            dir_name = str(int(time.time())) + str(random.randint(1, 3000)).zfill(4)

            data_dir = os.path.join(_base_path, config['DEFAULTS']['SAVE-DIR'], dir_name)

            while os.path.exists(data_dir):
                dir_name = str(int(time.time())) + str(random.randint(1, 3000)).zfill(4)
                data_dir = os.path.join(_base_path, config['DEFAULTS']['SAVE-DIR'], dir_name)

            os.makedirs(data_dir)

            logging.info('Directory %s created' % dir_name)

            for file in data:
                filename = secure_filename(file.filename)
                file.save(os.path.join(data_dir, filename))

            if data_dir:
                response = predictor.predict(data_dir)

                response = jsonify(response)

                response.headers.add('Access-Control-Allow-Origin', '*')  # Allow cross-origin request

                return response

    return jsonify({'error': 'Invalid request. No image attached'}), 422


if __name__ == '__main__':

    application.run(host='0.0.0.0', port=3031, debug=True)
