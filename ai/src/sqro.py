# --------------------------------------------------------------------------
# AI 모델 핵심 로직들을 호출하는 엔드포인트 모듈입니다.
# --------------------------------------------------------------------------
from dotenv import load_dotenv
from keras import optimizers

from tensorflow.test import is_gpu_available
from tensorflow.config.experimental import list_physical_devices, set_memory_growth

from ai.src.core.settings import GeneralSettings, TransformerSettings
from ai.src.model.model import MultiModalBert
from ai.src.utils.logging import setup_logging
from ai.src.core.exceptions import BackendException


def run_backend(settings) -> None:
    """
    main routine에서 호출하는 피싱 사이트 탐지기 엔드포인트.

    :return:
    """
    # Create the model
    # X_train_url, X_train_text, y_train = preprocess_data()  # TODO : impl this at /preprocess
    model = MultiModalBert(settings=settings)

    # 컴파일 및 학습
    model.compile(
        optimizer=optimizers.Adam(learning_rate=settings.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    # model.fit([X_train_url, X_train_text], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)


def process_parallel(settings: GeneralSettings):
    # TODO : impl GPU-distributed processing
    logger.info("Parallel option detected, fetching devices...")
    print("GPU available status:", is_gpu_available())
    gpus = list_physical_devices("GPU")
    if gpus:
        try:
            set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            raise BackendException(e)


if __name__ == "__main__":
    load_dotenv()
    general_settings = GeneralSettings()
    model_settings = TransformerSettings()

    logger = setup_logging(settings=general_settings)

    if general_settings.parallel is True:
        logger.debug("Checking parallel processing...")
        process_parallel(settings=general_settings)

    run_backend(model_settings)
