# --------------------------------------------------------------------------
# AI 모델 핵심 로직들을 호출하는 엔드포인트 모듈입니다.
# --------------------------------------------------------------------------
from dotenv import load_dotenv
from keras import optimizers
from tensorflow.test import is_gpu_available
from tensorflow.config.experimental import list_physical_devices, set_memory_growth

from core.settings import GeneralSettings, TransformerSettings
from model.model import MultiModalBert
from utils.logging import setup_logging
from core.exceptions import BackendException
from preprocess.dataset import DataConnector, PreProcessor


def preprocess_data(connector: DataConnector, processor: PreProcessor):
    """
    데이터베이스에서 데이터를 가져와 URL과 HTML 텍스트를 추출하고,
    이를 BERT 모델에 입력할 수 있도록 전처리합니다.
    """
    df = connector.get_filtered_data()
    url_inputs, text_inputs, labels = [], [], []

    for index, row in df.iterrows():
        urls, text = processor.preprocess_text(row["html_content"])
        url_inputs.append(urls)
        text_inputs.append(text)
        labels.append(row["label"])  # 라벨 필드

    return url_inputs, text_inputs, labels


def run_backend(settings) -> None:
    """
    main routine에서 호출하는 피싱 사이트 탐지기 엔드포인트.
    """
    # Data preparation
    connector = DataConnector(settings)
    processor = PreProcessor(settings)
    X_train_url, X_train_text, y_train = preprocess_data(
        connector, processor
    )  # noqa: N806

    # Create the model
    model = MultiModalBert(settings=settings)

    # 컴파일 및 학습
    model.compile(
        optimizer=optimizers.Adam(learning_rate=settings.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 모델 학습
    model.fit(
        [X_train_url, X_train_text],
        y_train,
        batch_size=settings.batch_size,
        epochs=settings.epoch,
        validation_split=0.2,
    )

    model.evaluate()


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

    if general_settings.parallel:
        logger.debug("Checking parallel processing...")
        process_parallel(settings=general_settings)

    run_backend(model_settings)
