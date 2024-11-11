# --------------------------------------------------------------------------
# 커스텀 임베딩 레이어 설정을 위한 모듈입니다.
# --------------------------------------------------------------------------
from keras.api.layers import Embedding, Dropout, Layer
from ai.src.core.settings import TransformerSettings


class CustomEmbedding(Layer):
    def __init__(self, settings: TransformerSettings):
        super(CustomEmbedding, self).__init__()
        self.embedding = Embedding(
            input_dim=settings.vocab_size,
            output_dim=settings.embedding_dim,
            trainable=True,
        )
        self.dropout = Dropout(settings.embedding_dropout)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout(x)
        return x
