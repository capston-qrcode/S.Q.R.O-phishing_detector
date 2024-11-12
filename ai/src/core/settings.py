# --------------------------------------------------------------------------
# Transformer 및 여러 Configuration, Settings 값을 설정하는 모듈입니다.
# --------------------------------------------------------------------------
import os
import logging


class GeneralSettings:
    """
    AI module general 설정
    """

    LOGGING_LEVEL: int = logging.DEBUG
    WORKSPACE: str = ""

    # 런타임 Config
    parallel: bool = False
    use_gpu: bool = False
    worker: int = 1

    @classmethod
    def __init__(cls) -> None:
        cls.WORKSPACE = os.getcwd()


class TransformerSettings(GeneralSettings):
    """
    Transformer 설정
    """

    # DB_PATH = os.path.abspath(
    #     os.path.join(
    #         os.path.dirname(__file__),
    #         "../preprocess/datas/test_database.db",
    #     )
    # )

    # 논문 참고 값
    EMBEDDING_DIM = 768
    MAX_TOKEN_LEN = 512
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 8
    EPOCHS = 10

    @classmethod
    def __init__(cls) -> None:
        super().__init__()


class ConverterSettings(GeneralSettings):
    """
    PreProcessor & HTML2Text converter 설정
    """

    # TODO : add me

    @classmethod
    def __init__(cls) -> None:
        super().__init__()
