import os.path
import shutil

import pandas as pd
from glob import glob

from parso.python.tree import String

from preprocessor.image import ImageProcessor
from scripts.utils import *

allowed_formats = ('jpg', 'img', )


class Preprocessor:
    def __init__(
            self,
            data_dir,
            text_reader_url = 'http://127.0.0.1:3030/transcript',
            max_lang = 2

    ):

        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.image_processor = ImageProcessor(text_reader_url,
            max_lang=max_lang)

    @staticmethod
    def check_and_download_image(image_url, save_path):
        try:
            # Send a GET request to the URL
            response = requests.get(image_url)

            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))

                image.save(save_path)
                logging.info(f"Image downloaded and saved as {save_path}")

                return save_path
            else:
                logging.info(f"Failed to retrieve the image. Status code: {response.status_code}")


        except Exception as error:
            logging.error(f"An error occurred: {error}")

        return ''

    def process_off_data(self, csv_file_path, sample_count=1000000):

        csv_basename = os.path.basename(csv_file_path).replace('.csv', '')

        try:

            tic = time.time()
            # Loading dataset
            with open(csv_file_path, 'rb') as file:
                lines = file.readlines()

            lines_to_be_processed = lines[1:sample_count] if sample_count else lines

            column_names = lines[0].decode("utf-8").replace('\n', '').split('\t')
            logging.info(f'columns headers : {column_names}')

            processed_data = []
            image_identifiers = config['OFF']['IDENTIFIER-COLUMNS']
            required_columns = config['OFF']['REQUIRED-COLUMNS']

            required_indices = [column_names.index(col) for col in image_identifiers]

            # Data Split and removing unwanted information from data

            for _, line in enumerate(lines_to_be_processed):
                line = line.decode("utf-8")
                line = line.replace('\n', '').split('\t')
                if len(line) == len(column_names) and all(line[idx] for idx in required_indices):
                    processed_data.append(line)

            data_df = pd.DataFrame(processed_data, columns=column_names)
            data_df.drop_duplicates(inplace=True)
            data_df = data_df[data_df[image_identifiers].apply(lambda x: all(x), axis=1)]

            # Handle missing data

            selected_columns = get_empty_columns(data_df, required_columns)

            data_df = data_df[list(selected_columns)].copy()

            for column in selected_columns - set(image_identifiers):
                data_df[column] = data_df[column].apply(lambda x: float(x) if x != '' else 0)

            # check the url and download images from off site

            data_df['image_path'] = data_df.apply(
                lambda x: self.check_and_download_image(
                    x['image_nutrition_url'], os.path.join(self.image_dir, x['code'] + '.jpg')),
                axis=1)

            data_df = data_df[data_df['image_path'] != '']

            if not data_df.empty:
                data_df.reset_index(inplace=True, drop=True)
                data_df.to_feather(os.path.join(self.data_dir, f'processed_data_{csv_basename}.feather'))

            toc = time.time()
            logging.info(f'execution time for processing {csv_basename} : {toc - tic}s')

            return data_df

        except Exception as error:
            logging.error(f'Off data {os.path.basename(csv_basename)} processing failed  {error}')

        return pd.DataFrame()


    def get_train_dataset(self, train_validation_split=0.9):
        train_dataset, validation_dataset = [[], []]

        entire_dataset = []

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        for csv_file in glob(self.data_dir + '/*.csv'):
            data_df = self.process_off_data(csv_file)
            entire_dataset.append(data_df)

        entire_dataset = pd.concat(entire_dataset) if entire_dataset else pd.DataFrame()

        if not entire_dataset.empty:

            total_images = os.listdir(self.image_dir)
            total_iterations = len(total_images)//10 + 1 if len(total_images)%10>=1 else len(total_images)//10

            temp_dir = os.path.join(self.image_dir, 'temp')

            bbox_df = []

            for i in range(total_iterations):
                init, final = (i*10, (i+1)*10)
                for img in total_images[init:final]:
                    shutil.copy(os.path.join(self.image_dir, img), os.path.join(temp_dir, img))

                _bbox_df = self.image_processor.image_preprocessing(self.image_dir)

                bbox_df.append(_bbox_df)

            bbox_df = pd.concat(bbox_df) if bbox_df else pd.DataFrame()

            if bbox_df.empty:
                logging.info('No image has text segments with in it')
                return

            entire_dataset = entire_dataset[entire_dataset['code'].isin(bbox_df['image_id'].unique())]



        return train_dataset, validation_dataset

    def preprocess_data(self, input_data):

        preprocessed_data = input_data

        return preprocessed_data



if __name__ == '__main__':

    import time

    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    _retina_url = 'http://10.20.0.15:3030/detect'
    pp = Preprocessor(_retina_url)
    _invoice_dir = r'D:\Facts and Figures\Nielsen\data'

    tic = time.time()

    _images, _bbox_df, _error = pp.image_preprocessing(_invoice_dir,)
    # _images, _languages = pp.convert_to_images(_invoice_dir)
    toc = time.time() - tic
    print(f'execution time {toc}')
    print('saving output...')
    # _bbox_df.reset_index(inplace=True, drop=True)
    # _bbox_df.to_feather(os.path.join(_invoice_dir, 'bbox_df'))
    _bbox_df.to_csv(os.path.join(_invoice_dir, 'bbox_df.csv'))
