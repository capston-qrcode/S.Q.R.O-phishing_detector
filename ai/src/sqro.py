# --------------------------------------------------------------------------
# 멀티 모달 기반 핵심 로직들을 호출하는 엔드포인트 모듈입니다.
# --------------------------------------------------------------------------
from keras.api import Model

from ai.src.transformer.model import create_multimodal_model
from core.settings import TransformerSettings


def _read_model_configs() -> None:
    # TransformerSettings() ~~ 로직 구현
    pass


def _clean_keras_backend() -> None:
    pass


def _refresh_keras_backend() -> None:
    pass


def run_keras_backend() -> None:
    """
    main routine에서 호출하는 피싱 사이트 탐지기 엔드포인트.

    :return:
    """
    # Load and preprocess the dataset
    _read_model_configs()
    # X_train_url, X_train_text, y_train = preprocess_data()  # TODO : impl this at /preprocess

    # Create the model
    model = create_multimodal_model()

    # Train the model
    # model.fit([X_train_url, X_train_text], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)


if __name__ == "__main__":
    _clean_keras_backend()
    run_keras_backend()
