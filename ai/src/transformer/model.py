# --------------------------------------------------------------------------
# BERT 모델을 선언하는 모듈입니다.
# --------------------------------------------------------------------------
from transformers import TFBertModel
from keras.api.layers import Input, Dense, Dropout, Concatenate
from keras.api.models import Model


class MultiHeadSelfAttention:
    """
    Attention head impl class
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


def finetune_bert() -> None:
    pass


def create_multimodal_model() -> Model:
    # TODO : input settings param & connect it
    url_input = Input(shape=(512,), dtype='int32', name='url_input')
    url_model = TFBertModel.from_pretrained('bert-base-uncased')
    url_output = url_model(url_input)[1]

    text_input = Input(shape=(512,), dtype='int32', name='text_input')
    text_model = TFBertModel.from_pretrained('bert-base-uncased')
    text_output = text_model(text_input)[1]

    concatenated = Concatenate()([url_output, text_output])
    x = Dense(768, activation='gelu')(concatenated)
    x = Dropout(0.1)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[url_input, text_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
