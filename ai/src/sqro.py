# --------------------------------------------------------------------------
# AI 모델 핵심 로직들을 호출하는 엔드포인트 모듈입니다.
# --------------------------------------------------------------------------
import keras
import tensorflow as tf

from dotenv import load_dotenv
from keras.src.optimizers import Adam
from keras.src.optimizers.schedules import PolynomialDecay
from tensorflow.test import is_gpu_available
from tensorflow.config.experimental import list_physical_devices, set_memory_growth

from core.settings import GeneralSettings, TransformerSettings
from model.model import MultiModalBert
from utils.logging import setup_logging
from core.exceptions import BackendException
from model.evaluate import F1Score
from preprocess.dataset import DataConnector, PreProcessor


def preprocess_data(connector: DataConnector, processor: PreProcessor):
    df = connector.get_filtered_data()
    url_inputs, text_inputs, url_masks, text_masks, labels = [], [], [], [], []

    for _, row in df.iterrows():
        url_encoded, url_mask = processor.preprocess_url(row["url"])
        text_encoded, text_mask = processor.preprocess_text(row["html_content"])

        # 둘 중 하나라도 데이터가 없으면 무시
        if url_encoded.shape[0] == 0 or text_encoded.shape[0] == 0:
            continue

        url_inputs.append(url_encoded)
        text_inputs.append(text_encoded)
        url_masks.append(url_mask)
        text_masks.append(text_mask)
        labels.append(row["label"])

    url_inputs = tf.stack(url_inputs)
    text_inputs = tf.stack(text_inputs)
    url_masks = tf.stack(url_masks)
    text_masks = tf.stack(text_masks)
    labels = tf.convert_to_tensor(labels)
    return url_inputs, url_masks, text_inputs, text_masks, labels


def run_backend(settings) -> None:
    """
    main routine에서 호출하는 피싱 사이트 탐지기 엔드포인트.
    """
    connector = DataConnector(settings)
    processor = PreProcessor()
    X_train_url, X_train_url_mask, X_train_text, X_train_text_mask, y_train = preprocess_data(connector, processor)

    model = MultiModalBert(settings=settings)

    lr_schedule = PolynomialDecay(
        initial_learning_rate=settings.learning_rate,
        decay_steps=10000,
        end_learning_rate=1e-7
    )
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", F1Score()]
    )

    model.fit(
        [X_train_url, X_train_url_mask, X_train_text, X_train_text_mask],
        y_train,
        batch_size=settings.batch_size,
        epochs=settings.epoch,
        validation_split=0.2
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
