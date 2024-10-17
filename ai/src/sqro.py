# --------------------------------------------------------------------------
# 멀티 모달 기반 핵심 로직들을 호출하는 엔드포인트 모듈입니다.
# --------------------------------------------------------------------------
from keras.api import Model

from core.settings import TransformerSettings


def _read_model_configs() -> None:
    # TransformerSettings() ~~ 로직 구현
    pass


def _clean_keras_backend() -> None:
    pass


def _refresh_keras_backend() -> None:
    pass


def create_transformer() -> Model:
    pass


def run_keras_backend() -> None:
    """
    main routine에서 호출하는 피싱 사이트 탐지기 엔드포인트.

    :return:
    """
    pass


if __name__ == "__main__":
    run_keras_backend()
