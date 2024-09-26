import os
import logging
import configparser
from difflib import SequenceMatcher
import pytesseract
from config.config import config
import requests
from PIL import Image
from io import BytesIO


def get_empty_columns(data_df, required_columns):

    selected_columns = []

    if isinstance(data_df, pd.DataFrame) and not data_df.empty:

        data_describe = data_df.describe()

        _temp_describe = data_describe.loc['unique', :]
        empty_columns = _temp_describe[_temp_describe == 1].index
        selected_columns = set(required_columns) - set(empty_columns)

    return selected_columns


def delete_all_files(directory):

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            os.remove(file_path)  # Delete the file

def flatten_list(item):
    container = []
    if isinstance(item, (tuple, list)):
        for entity in item:
            if not isinstance(entity, (tuple, list)):
                container.append(entity)
            else:
                container += flatten_list(entity)
    else:
        container = [item]

    return container


def get_unique(raw_list):
    unique_values = []
    for item in raw_list:
        if item not in unique_values:
            unique_values.append(item)
    return unique_values


def create_request(
        url, files, headers=None, payload=None
):
    if not headers:
        headers = {}
    if not payload:
        payload = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    return response.content


def get_content_type(ext):
    ext = ext.replace('.', '')

    content_type = ''
    if ext in ('jpg', 'jpeg', 'JPG', 'JPEG'):
        content_type = 'image/jpeg'

    elif ext in ('png', 'PNG'):
        content_type = 'image/png'

    elif ext in ('tsv', 'TSV'):
        content_type = 'application/octet-stream'

    return content_type


def validate_page(df, ):
    validity = True

    if not df.empty:

        alpha_df = df[df['easyocr_transcript'].notnull()].copy()
        alpha_df['word_count'] = alpha_df['easyocr_transcript'].apply(
            lambda x: len([_item for _item in x.split() if not any(char.isdigit() for char in _item)])
        )

        word_count = sum(alpha_df['word_count'])
        if word_count >= 200:
            return False

    return validity


def service_manager_loader():

    service_manager = configparser.ConfigParser()
    if os.path.exists(config['SERVICE-MANAGER']['GLOBAL']):
        service_manager.read(config['SERVICE-MANAGER']['GLOBAL'])
    else:
        service_manager.read(config['SERVICE-MANAGER']['LOCAL'])

    return service_manager


def get_orientation(img):
    rotate_angle = 0
    rotate_confidence = 0
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        rotate_angle = osd['rotate']
        rotate_confidence = float(osd['orientation_conf'])

    except Exception as error:
        logging.error('Orientation calculation failed %s' % error)
    return rotate_angle, rotate_confidence
