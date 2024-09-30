import torch
from transformers import BertTokenizer, BertModel
from config.config import config
from scripts.utils import flatten_list
from preprocessor.preprocess import Preprocessor
from model.entity_tagging_model import EntityTaggingModel


class Predictor:
    def __init__(self):

        nutrition_columns = list('_'.join(label.split('_')[:2]) for label in config['DATA']['NUTRITION-COLUMNS'])
        total_labels = flatten_list([[prefix + label for prefix in ('B-', 'I-')] for label in nutrition_columns])
        tag2idx = {label: i + 1 for i, label in enumerate(total_labels)}
        tag2idx['O'] = 0

        self.idx2tag = {value: key for key, value in tag2idx.items()}

        self.model = EntityTaggingModel(bert_model=BertModel.from_pretrained('bert-base-uncased'), num_tags=len(tag2idx))

        # Load the saved weights
        self.model.load_state_dict(torch.load(config['MODEL']['']))
        self.model.eval()

        # Initialize the tokenizer (should be the same as during training)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def predict_entities(self, text):

        tokens = self.tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']

        # Ensure no gradients are being calculated during inference
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)

        # Get the predicted tag indices
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

        # Convert indices to tags
        predicted_tags = [self.idx2tag[pred] for pred in predictions[:len(text.split())]]  # Ignoring padding tokens

        return predicted_tags


    # Function for inference
    def predict(self, data_dir):
        preprocessor = Preprocessor(data_dir)

        prediction_output = {}

        bbox_df = preprocessor.preprocess_data()
        if not bbox_df.empty:
            bbox_df['tags'] = bbox_df['transcript'].apply(lambda x: self.predict_entities(x))
            for image in bbox_df['image_id'].unique():
                tags = flatten_list(bbox_df[bbox_df['image']==image]['tags'].tolist())
                prediction_output[image] = tags

        return prediction_output
