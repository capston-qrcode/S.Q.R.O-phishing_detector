# --------------------------------------------------------------------------
# BERT 모델을 선언하는 모듈입니다.
# --------------------------------------------------------------------------
from keras import Model
from transformers import TFBertModel
from keras.api.layers import Input, Dense, Dropout, Concatenate

from ai.src.utils.logging import setup_logging
from ai.src.core.settings import TransformerSettings


class MultiModalBert(Model):
    def __init__(self, settings: TransformerSettings):
        super(MultiModalBert, self).__init__()
        self.html_bert = TFBertModel.from_pretrained("bert-base-uncased")
        self.url_bert = TFBertModel.from_pretrained("bert-base-uncased")

        # Fully Connected Layer 및 드롭아웃
        self.fc = Dense(settings.embedding_dim * 2, activation="gelu")  # 논문 기준, 2 * 768 크기로 확장
        self.dropout = Dropout(settings.embedding_dropout)
        self.classifier = Dense(1, activation="sigmoid")

    def call(self, url_input, url_mask, text_input, text_mask):
        url_output = self.url_bert(url_input, attention_mask=url_mask)[1]
        html_output = self.html_bert(text_input, attention_mask=text_mask)[1]

        # BERT 출력 결합
        combined_output = Concatenate()([url_output, html_output])
        x = self.fc(combined_output)
        x = self.dropout(x)
        return self.classifier(x)
