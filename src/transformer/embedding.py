# --------------------------------------------------------------------------
# 임베딩을 설정하는 모듈입니다.
# --------------------------------------------------------------------------
import numpy as np

from keras.api.layers import Layer


def _get_matrix() -> np.array:
    result = np.array([])

    return result


class Embedding(Layer):
    """
    Layer impl class
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        pass
