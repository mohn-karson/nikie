import os
import json
import logging
import base64
import cv2
from glob import glob
import pandas as pd
from PIL import Image
from io import BytesIO
# import pytesseract
import requests
from scripts import language_detector
from scripts.utils import create_request, validate_page


allowed_formats = ('jpg', 'img', )

class ImageProcessor:
    def __init__(
            self,
            text_reader_url,
            max_lang=2
    ):
        self.text_reader_url = text_reader_url
        self.max_lang = max_lang


    def validate_images(self, data_dir):

        images = []

        for idx, image in enumerate(glob(data_dir+ '/*')):

            doc_extension = os.path.basename(image).split('.')[-1]
            image_name = os.path.basename(image).replace(doc_extension, '')

            try:

                if doc_extension in allowed_formats:

                    img = cv2.imread(image)

                    # Language detection
                    languages = language_detector.lang_detect(image=img, max_lang=self.max_lang)
                    languages = languages if languages else [('en', 'eng'), ]
                    logging.info(f'{image_name} : Detected Languages: {languages}')

                    # Image preprocessing - rotation and skew handling
                    encoded_image = base64.b64encode(open(image, 'rb').read()).decode('utf-8')

                    images.append({
                        'imageName': image_name,
                        'encodedImage': encoded_image,
                        'languages': languages,
                    })
            except Exception as error:
                logging.error(f'Invalid Image {image_name} - {error}')

        return images

    def image_preprocessing(
            self, image_dir,
    ) :

        bbox_df = []
        error = ''

        images = self.validate_images(image_dir)

        try:

            if images:
                logging.info(f'Images to be processed {[image["imageName"] for image in images]}')

                text_reader_input = {
                    'images': images,
                }
                text_reader_payload = json.dumps(text_reader_input)
                headers = {'Content-Type': 'application/json'}

                retina_output = create_request(self.text_reader_url, files=None, payload=text_reader_payload, headers=headers)
                bboxes = json.loads(retina_output)

                # Detects Language
                for idx, image in enumerate(images):
                    image_id = image['imageName']

                    retina_df = pd.DataFrame(json.loads(bboxes[image_id]))

                    if validate_page(df=retina_df, ):

                        retina_df['transcript'] = retina_df['easyocr_transcript'].apply(
                            lambda x: x.strip())

                        #     retina_df['transcript'] = retina_df[['xmin', 'xmax', 'ymin', 'ymax']].apply(
                        #         lambda x: self.tesseract_plus(
                        #             image=img, xmin=x['xmin'], ymin=x['ymin'], xmax=x['xmax'], ymax=x['ymax'],
                        #             languages=alpha3_languages), axis=1
                        #     )

                        # retina_df['transcript'] = retina_df[['easyocr_transcript', 'transcript']].apply(
                        #     lambda x: x['transcript'].strip() if x['transcript']
                        #     else x['easyocr_transcript'].strip(), axis=1)

                    retina_df['transcript'] = retina_df[['easyocr_transcript', 'transcript']].apply(
                        lambda x: x['transcript'] if len(x['easyocr_transcript'])//max(len(x['transcript']), 1) < 2
                        else x['easyocr_transcript'], axis=1)

                    retina_df.loc[:, 'image'] = image_id

                    image['boundingBox'] = retina_df.to_json(orient='records')
                    images[image_id] = image

                    bbox_df.append(retina_df)

            bbox_df = pd.concat(bbox_df) if bbox_df else pd.DataFrame()

        except Exception as err:
            error = err
            bbox_df = pd.DataFrame()
            logging.error('Image processing failed: {}'.format(error))

        return images, bbox_df, error
