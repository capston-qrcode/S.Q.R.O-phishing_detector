# --------------------------------------------------------------------------
# 영어로된 데이터만 필터링하는 모듈입니다.
# --------------------------------------------------------------------------
from langdetect import detect, LangDetectException
from typing import Optional


def filter_english(content: str) -> Optional[str]:
    """
    영어로 된 텍스트만 반환하고, 그렇지 않으면 None을 반환하는 함수입니다.

    :param content: 필터링할 텍스트 콘텐츠
    :return: 영어 텍스트 또는 None
    """
    try:
        if detect(content) == "en":
            return content
    except LangDetectException:
        pass
    return None
