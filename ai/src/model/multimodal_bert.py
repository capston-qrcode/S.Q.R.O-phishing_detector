import torch
from torch import nn
from transformers import BertModel


class MultimodalBERT(nn.Module):
    def __init__(self, embedding_dim=768):
        super(MultimodalBERT, self).__init__()
        self.url_bert = BertModel.from_pretrained("bert-base-uncased")
        self.html_bert = BertModel.from_pretrained("bert-base-uncased")

        # concat layer
        self.fc = nn.Linear(embedding_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(
        self, url_input_ids, url_attention_mask, html_input_ids, html_attention_mask
    ):
        # pass URL & HTML to BERT model
        url_output = self.url_bert(
            input_ids=url_input_ids, attention_mask=url_attention_mask
        )
        html_output = self.html_bert(
            input_ids=html_input_ids, attention_mask=html_attention_mask
        )

        # combine CLS token output
        combined_output = torch.cat(
            (url_output.pooler_output, html_output.pooler_output), dim=1
        )

        # Fully Connected Layer & classifier
        x = self.fc(combined_output)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits
