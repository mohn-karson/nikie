import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertModel
import torch
from torch import nn
from torch.utils.data import DataLoader

from config.config import config
from preprocessor.preprocess import Preprocessor
from preprocessor.entity_tagging_dataset import EntityTaggingDataset
from model.entity_tagging_model import EntityTaggingModel
from scripts.utils import *


def train_model(data_dir, num_epochs=100, split_ratio=0.9):
    preprocessor = Preprocessor(data_dir, text_reader_url='http://127.0.0.1:3030/transcript')
    dataset_path = os.path.join(data_dir, 'entity_dataset.pickle')
    if not os.path.exists(dataset_path):
        preprocessor.create_training_dataset(dataset_name='entity_dataset')

    entire_dataset = pd.read_pickle(dataset_path)

    nutrition_columns = list('_'.join(label.split('_')[:2]) for label in config['DATA']['NUTRITION-COLUMNS'])
    total_labels = flatten_list([[prefix + label for prefix in ('B-', 'I-')] for label in nutrition_columns])
    tag2idx = {label: i + 1 for i, label in enumerate(total_labels)}
    tag2idx['O'] = 0

    train_df, val_test_df = train_test_split(entire_dataset, test_size=0.1)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5)

    train_dataset = EntityTaggingDataset(train_df, tag2idx)
    val_dataset = EntityTaggingDataset(val_df, tag2idx)
    test_dataset = EntityTaggingDataset(test_df, tag2idx)

    # Create DataLoaders for each dataset
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    model = EntityTaggingModel(bert_model=BertModel.from_pretrained('bert-base-uncased'), num_tags=len(tag2idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop with validation
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # Training step
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            entity_tags = batch['entity_tags']

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            # Calculate loss
            loss = loss_fn(outputs.view(-1, len(tag2idx)), entity_tags.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                entity_tags = batch['entity_tags']

                outputs = model(input_ids, attention_mask)
                val_loss = loss_fn(outputs.view(-1, len(tag2idx)), entity_tags.view(-1))
                total_val_loss += val_loss.item()

        print(f"Epoch {epoch + 1}, Training Loss: {total_train_loss}, Validation Loss: {total_val_loss}")
