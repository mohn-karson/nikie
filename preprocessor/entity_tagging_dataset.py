import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config.config import config
from scripts.utils import flatten_list


class EntityTaggingDataset(Dataset):
    def __init__(self, data_df, tag2idx, tokenizer_name='bert-base-uncased', max_len=128):

        self.data_df = data_df
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.tag2idx = tag2idx

        self.transcripts = []
        self.annotations = []

        for code in data_df['code'].unique():
            df = data_df[data_df['code']==code]
            self.transcripts.append(df['transcript'].tolist())
            self.annotations.append(df['annotations'].tolist())

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx):
        # Load transcript
        transcript = self.transcripts[idx]
        annotations = self.annotations[idx]

        # Tokenize transcript
        tokens = self.tokenizer(transcript, padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")

        entity_tags = self.prepare_entity_tags(transcript, annotations)

        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'token_type_ids': tokens.get('token_type_ids', None),
            'entity_tags': torch.tensor(entity_tags)
        }

    def prepare_entity_tags(self, transcript, annotations):
        words = transcript.split()

        entity_tags = ['O'] * len(words)

        for line, entities in annotations.items():
            for entity in entities:
                word = entity['word']
                tag = entity['tag']

                for i, w in enumerate(words):
                    if w == word and entity_tags[i] == 'O':
                        entity_tags[i] = tag
                        break

        return [self.tag2idx[tag] for tag in entity_tags]