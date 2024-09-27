import os
import logging
import pytesseract
import langdetect
from PIL import Image
from scripts.languages import languages_alpha2to3


def lang_detect(image, img_path=None, max_lang=2):
    detected_languages = list()

    try:
        txt = ''
        if not img_path:
            txt = pytesseract.image_to_string(image, config='--psm 6')

        elif os.path.exists(img_path):
            file_format = img_path.split(os.path.sep)[-1].split('.')[-1]

            if file_format in ['png', 'jpg', 'jpeg']:
                txt = pytesseract.image_to_string(Image.open(img_path), lang='eng', config='--psm 6')

        if txt:
            languages = langdetect.detect_langs(txt)
            for prb in languages:
                if prb.lang in languages_alpha2to3.keys():
                    detected_languages.append((prb.lang, languages_alpha2to3[prb.lang]))

    except Exception as err:
        logging.error(err)
        detected_languages = []

    # logging.info('Detected Languages: {}'.format(detected_langs))

    detected_languages = detected_languages if detected_languages else [('en', 'eng'), ]
    return detected_languages[:max_lang]

