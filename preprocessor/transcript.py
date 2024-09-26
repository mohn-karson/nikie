from difflib import SequenceMatcher
from deep_translator import GoogleTranslator

from config.config import config

class TextProcessor:

    @staticmethod
    def translate(text_segment, language='auto'):

        translated = ''

        if text_segment:
            translator = GoogleTranslator(source=language, target='en')
            translated = translator.translate(text_segment)

        return translated

    @staticmethod
    def annotate_beginning(text_segment, labels):
        tokens = text_segment.split()  # Tokenize the text segment
        annotation = []
        identified_label = ''
        labelled_tokens = []

        for label in labels:
            label_len = min(len(label), 2)
            token_len = len(tokens)
            if token_len < label_len:
                continue
            for i in range(0, token_len-label_len+1):
                selected_tokens = tokens[i*label_len: (i+1)*label_len]
                token_combined = ' '.join(selected_tokens)
                if SequenceMatcher(None, token_combined, label).ratio() > 0.6:
                    annotation = [
                        {'word': token, 'tag': 'B-' + label} for token in selected_tokens]
                    labelled_tokens += selected_tokens
                    identified_label = label
                    break
            if identified_label:
                break

        annotation += [{'word': _token, 'tag': 'O'} for _token in set(tokens) - set(labelled_tokens)]

        return annotation, identified_label

    @staticmethod
    def annotate_inside(text_segment, label, quantity):
        tokens = text_segment.split()

        return [{'word': token, 'tag': 'I-' + label} for token in tokens if str(quantity) in token]

    def text_processing(self, data_df, bbox_df):

        final_annotations = {}

        nutrition_columns = set(data_df.columns) - set(config['DATA']['IDENTIFIER-COLUMNS'])
        data_df.set_index('code', inplace=True, drop=True)


        for code, value in data_df['code'].iterrows():

            df = bbox_df[bbox_df['image_id']==code].copy()
            df['translated_transcript'] = df[['transcript', 'language']].apply(
                lambda x: self.translate(text_segment=x['transcript'], language=x['language']) if x['language']!='en'
                else x['transcript'], axis=1
            )

            nutrition_info = value[nutrition_columns]

            possible_labels = [' '.join(label.split(' ')[:2]) for label in nutrition_info[nutrition_info>0].index]
            possible_labels_dict = {
                nutrition: str(quantity) for nutrition, quantity in nutrition_info[nutrition_info>0]
            }

            for row in df['row'].unique():
                temp_df = df[df['row']==row]

                identified_annotation=False
                for bidx, _value in temp_df.iterrows():
                    beginning_annotation, label = self.annotate_beginning(
                        value['translated_transcript'], possible_labels
                    )

                    if beginning_annotation:
                        for iidx in (bidx, bidx+1):

                            inside_annotation = self.annotate_inside(
                                temp_df.iloc[iidx, 'translated_transcript'], label, possible_labels_dict[label]
                            )
                            if inside_annotation:
                                possible_labels.remove(label)
                                final_annotations[_value['line_identifier']] = (
                                        beginning_annotation + inside_annotation)

                                identified_annotation = True
                                break

                    if  identified_annotation:
                        break

        return final_annotations


