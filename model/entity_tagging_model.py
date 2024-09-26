import torch.nn as nn

class EntityTaggingModel(nn.Module):
    def __init__(self, bert_model, num_tags):
        super(EntityTaggingModel, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = self.classifier(outputs.last_hidden_state)  # [batch_size, seq_len, num_tags]
        return logits
