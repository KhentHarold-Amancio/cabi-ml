import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')


def get_bert_embeddings(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Return the last hidden state of the BERT model
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
