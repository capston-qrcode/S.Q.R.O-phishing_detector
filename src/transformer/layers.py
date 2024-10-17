# --------------------------------------------------------------------------
# 모델 레이어를 설정하는 모듈입니다.
# --------------------------------------------------------------------------
from keras.api.layers import Layer


class Gelu(Layer):
    """
    Gelu function custom impl class
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
